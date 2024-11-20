import argparse
import logging

import numpy as np
import torch
from model import GeneralVae, PictureDecoder, PictureEncoder
from sklearn.linear_model import LinearRegression
from torchvision.utils import save_image

logger = logging.getLogger("cairosvg")
logger.setLevel(logging.CRITICAL)




if __name__ == "__main__":
    args = get_args()

    starting_epoch = 1
    total_epochs = None

    # seed = 42
    # torch.manual_seed(seed)

    log_interval = 25
    LR = 5.0e-4

    cuda = True
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

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


    times = int(args.n / args.b)
    print(
        f"Using batch size {args.b} and sampling {times} times for a total of {args.b * times} samples drawn. Saving {'images' if args.image else 'numpy array'}"
    )
    samples = []
    for i in range(times):
        with torch.no_grad():
            sample = torch.randn(args.b, 512).to(device)
            sample = model.decode(sample).cpu()

            if args.image:
                save_image(
                    sample.view(args.b, 3, 256, 256),
                    args.o + "sample_" + str(i) + ".png",
                )
            else:
                samples.append(sample.view(args.b, 3, 256, 256).cpu().numpy())

    if not args.image:
        samples = np.concatenate(samples, axis=0)
        np.save(f"{args.o}samples.npy", samples)