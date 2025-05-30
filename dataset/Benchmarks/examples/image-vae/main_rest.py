import argparse
import datetime
import itertools
import logging

import numpy as np
import pandas as pd
import torch
from dataloader import MoleLoader
from model import GeneralVae, PictureDecoder, PictureEncoder, customLoss
from sklearn.linear_model import LinearRegression
from torch import nn, optim
from torchvision.utils import save_image
from utils import AverageMeter

logger = logging.getLogger("cairosvg")
logger.setLevel(logging.CRITICAL)








if __name__ == "__main__":
    args = get_args()

    starting_epoch = 1
    total_epochs = None

    seed = 42
    torch.manual_seed(seed)

    log_interval = 25
    LR = 5.0e-4

    output_dir = args.op
    save_files = args.mp

    data_para = True
    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")
    kwargs = (
        {"num_workers": args.workers, "pin_memory": True}
        if cuda
        else {"num_workers": args.workers}
    )

    print("\nloading data...")
    smiles_lookup_train = pd.read_csv(f"{args.d}/train.csv")
    print(smiles_lookup_train.head())
    smiles_lookup_test = pd.read_csv(f"{args.d}/test.csv")
    print(smiles_lookup_test.head())
    print("Done.\n")

    encoder = PictureEncoder(rep_size=512)
    decoder = PictureDecoder(rep_size=512)

    checkpoint = None
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        print(
            f"Loading Checkpoint ({args.checkpoint}). Starting at epoch: {checkpoint['epoch'] + 1}."
        )
        starting_epoch = checkpoint["epoch"] + 1
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        decoder.load_state_dict(checkpoint["decoder_state_dict"])

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    model = GeneralVae(encoder, decoder, rep_size=512).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    if checkpoint is not None:
        print("using optimizer past state")
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = LR
    print("LR: {}".format(LR))

    loss_picture = customLoss()

    if data_para and torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        loss_picture = nn.DataParallel(loss_picture)

    val_losses = []
    train_losses = []

    train_data = MoleLoader(smiles_lookup_train)
    train_loader_food = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs
    )

    val_data = MoleLoader(smiles_lookup_test)
    val_loader_food = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs
    )




    if total_epochs is None:
        trn_rng = itertools.count(start=starting_epoch)
    else:
        trn_rng = range(starting_epoch, total_epochs)

    for epoch in trn_rng:
        for param_group in optimizer.param_groups:
            print("Current learning rate is: {}".format(param_group["lr"]))

        loss = train(epoch)
        test(epoch)

        torch.save(
            {
                "epoch": epoch,
                "encoder_state_dict": model.module.encoder.state_dict(),
                "decoder_state_dict": model.module.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_files + "epoch_" + str(epoch) + ".pt",
        )
        torch.save(model.module, "model_inf.pt")
        with torch.no_grad():
            sample = torch.randn(64, 512).to(device)
            sample = model.module.decode(sample).cpu()
            save_image(
                sample.view(64, 3, 256, 256),
                output_dir + "sample_" + str(epoch) + ".png",
            )