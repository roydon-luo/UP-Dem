import torch
import torch.nn as nn
import torch.nn.functional as F
from .SFI import TransformerBlock_P

# Feature Extractor Module
class ImgFeatureExtractorModule(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(ImgFeatureExtractorModule, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class PolarFeatureExtractorModule(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(PolarFeatureExtractorModule, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.SiLU(),
        )

    def forward(self, x):
        x = self.proj(x)

        return x

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

def get_padder(padding_mode='reflection',
               padding=1,
               value=None):
    if padding_mode.lower() not in ['reflection','replication','zero','zeros','constant']:
        raise Exception('Only support "reflection","replication","zero" or "constant". But get "%s."'%(padding_mode))
    if padding_mode.lower()=='reflection':
        return nn.ReflectionPad2d(padding)
    if padding_mode.lower()=='replication':
        return nn.ReplicationPad2d(padding)
    if padding_mode.lower() in ['zero', 'zeros']:
        return nn.ZeroPad2d(padding)
    if padding_mode.lower() in 'constant':
        value=0 if value==None else value
        return nn.ConstantPad2d(padding,value)

def calculate_polar(img):
    I0, I45, I90, I135 = img[:,0:3], img[:,3:6], img[:,6:9], img[:,9:12]
    S1 = (I0 - I90 + 1) / 2
    S2 = (I45 - I135 + 1) / 2
    return torch.cat([S1, S2], 1)

class Conv2d(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 dilation,
                 groups,
                 bias,
                 padding_mode='reflection',
                 padding='same',
                 value=None):
        super(Conv2d, self).__init__()
        padding = int(int(1+dilation*(kernel_size-1))//2) if padding=='same' else 0
        self.pad = get_padder(padding_mode, padding, value)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                0, dilation, groups, bias)
    def forward(self, code):
        return self.conv2d(self.pad(code))


class DictConv2d(nn.Module):
    def __init__(self, img_channels,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1, # only support stride=1
                 dilation=1,
                 groups=1,
                 bias=False,
                 num_convs=1,
                 padding_mode='reflection',
                 padding='same',
                 value=None
                 ):
        super(DictConv2d, self).__init__()
        self.in_channels = in_channels

        # decoder
        self.conv_decoder = nn.Sequential()
        self.conv_decoder.add_module('de_conv0',
                                     Conv2d(in_channels, img_channels, kernel_size, 1, dilation, groups, bias,
                                            padding_mode, padding, value))
        for i in range(1, num_convs):
            self.conv_decoder.add_module('de_conv' + str(i),
                                         Conv2d(img_channels, img_channels, kernel_size, 1, dilation, groups, bias,
                                                padding_mode, padding, value))

        # encoder
        self.conv_encoder = nn.Sequential()
        for i in range(num_convs - 1):
            self.conv_encoder.add_module('en_conv' + str(i),
                                         Conv2d(img_channels, img_channels, kernel_size, 1, dilation, groups, bias,
                                                padding_mode, padding, value))
        self.conv_encoder.add_module('en_conv' + str(num_convs - 1),
                                     Conv2d(img_channels, in_channels, kernel_size, 1, dilation, groups, bias,
                                            padding_mode, padding, value))

        self.shift_flag = out_channels != in_channels
        self.conv_channel_shift = nn.Conv2d(in_channels, out_channels, 1) if self.shift_flag else None

    def _forward(self, data, code):
        dcode = self.conv_decoder(code)

        if data.shape[2] != dcode.shape[2] or data.shape[3] != dcode.shape[3]:
            data = F.interpolate(data, size=dcode.shape[2:])
        res = data - dcode
        dres = self.conv_encoder(res)

        if code.shape[2] != dres.shape[2] or code.shape[3] != dres.shape[3]:
            code = F.interpolate(code, size=dres.shape[2:])
        code = code + dres

        if self.shift_flag:
            code = self.conv_channel_shift(code)
        return code

    def forward(self, data, code):
        return self._forward(data, code)


class DictConv2dBlock(nn.Module):
    def __init__(self, img_channels,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 num_convs=1, # the number of conv units in decoder and encoder
                 act='prelu',
                 act_value=None,
                 norm=True, # group_normalization (None will disable it)
                 padding_mode='reflection',
                 padding_value=None, # padding constant (Ignore it if padding_mode is not "constant")
                 ):
        super(DictConv2dBlock, self).__init__()
        self.conv = DictConv2d(img_channels,in_channels,out_channels, kernel_size, stride,
                   dilation, groups, bias, num_convs)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels) if norm!=None else None
        if act != 'prelu':
            raise ValueError(f"Unsupported activation: {act}")
        self.activation = nn.PReLU(out_channels, 0.25 if act_value is None else act_value)
    def _forward(self, data, code):
        code = self.conv(data, code)
        code = self.norm(code)
        code = self.activation(code)
        return data, code

    def forward(self, data, code):
        return self._forward(data, code)


class UPDem(nn.Module):
    def __init__(self, img_channels,
        in_channels,
        out_channels,
        kernel_size,
        ista_iters,
        stride=1,
        dilation=1,
        groups=1,
        bias=False,
        num_convs=3, # the number of conv units in decoder and encoder
        act='prelu',
        act_value=None,
        ):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)
        self.pdm_upsample = Upsample(in_channels*2)
        self.con1x1 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=bias)
        # Feature Extractor
        self.img_feature_extractor = ImgFeatureExtractorModule(img_channels, in_channels)
        self.polar_feature_extractor = PolarFeatureExtractorModule(6, in_channels)

        self.cdm_convLayer0 = DictConv2dBlock(
                img_channels,in_channels,out_channels, kernel_size, stride,
                dilation, groups, bias, num_convs,act,act_value)
        self.cdm_convLayer = nn.ModuleList(
                [DictConv2dBlock(
                        img_channels,in_channels,out_channels, kernel_size, stride,
                        dilation, groups, bias, num_convs,act,act_value)
                for i in range(ista_iters)])

        self.pdm_convLayer0 = DictConv2dBlock(
            img_channels, in_channels, out_channels, kernel_size, stride,
            dilation, groups, bias, num_convs, act, act_value)
        self.pdm_convLayer = nn.ModuleList(
            [DictConv2dBlock(
                img_channels, in_channels, out_channels, kernel_size, stride,
                dilation, groups, bias, num_convs, act, act_value)
                for i in range(ista_iters)])

        self.encoder = TransformerBlock_P(dim=in_channels,
                                     num_heads=8,
                                     ffn_expansion_factor=2,
                                     bias=False,
                                     LayerNorm_type='WithBias')

        self.cdm_decoder = TransformerBlock_P(dim=in_channels,
                                    num_heads=8,
                                    ffn_expansion_factor=2,
                                    bias=False,
                                    LayerNorm_type='WithBias')

        self.pdm_decoder = TransformerBlock_P(dim=2*in_channels,
                                              num_heads=8,
                                              ffn_expansion_factor=2,
                                              bias=False,
                                              LayerNorm_type='WithBias')

        self.mid_output = nn.Conv2d(in_channels, img_channels, 3, 1, 1, bias=bias)
        self.final_output = nn.Conv2d(in_channels, img_channels, 3, 1, 1, bias=bias)


    def forward(self, img_input):
        # color demosaicking
        polar = calculate_polar(img_input)
        polar_feat = self.polar_feature_extractor(polar)
        img_feat = self.img_feature_extractor(img_input)
        img_feat = self.encoder(img_feat, polar_feat)
        z_c = img_feat

        for layer in self.cdm_convLayer:
            img_input, z_c = layer(img_input, z_c)
        img_input, z_c = self.cdm_convLayer0(img_input, z_c)

        z_c = self.cdm_decoder(z_c, polar_feat)
        mid_out = self.mid_output(z_c) + img_input

        # polarization demosaicking
        polar = calculate_polar(mid_out)
        polar_feat = self.polar_feature_extractor(polar)
        z_p = z_c
        for layer in self.pdm_convLayer:
            img_input, z_p = layer(img_input, z_p)
        img_input, z_p = self.pdm_convLayer0(img_input, z_p)
        polar_feat2 = self.con1x1(polar_feat)
        z_p = torch.cat([z_p, z_c], 1)
        z_p = self.pdm_decoder(z_p, polar_feat2)
        img_out = self.final_output(self.pdm_upsample(z_p)) + F.interpolate(img_input, scale_factor=2,
                                                                                             mode='bilinear')

        return mid_out, img_out
