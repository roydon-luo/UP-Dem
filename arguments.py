"""Default architecture and paths for the published UP-Dem checkpoint."""


class args:
    num_input_channels = 12
    kernel_channel = 128
    kernel_size = 3
    ista_iters = 7

    train_path = "outputs/train"
    val_data_path = "data/paired"
    raw_data_path = "data/raw"
    test_path = "outputs/inference"
    checkpoint_path = "checkpoints/updem.pth"

    device = "cuda"
    lr = 1e-4
    img_size = 128
    train_batch_size = 24
    test_batch_size = 1
    epochs = 100
