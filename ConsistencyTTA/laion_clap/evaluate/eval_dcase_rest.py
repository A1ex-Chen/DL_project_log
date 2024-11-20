import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from open_clip import create_model
from open_clip import tokenize
import glob
import json
import librosa
from tqdm import tqdm
import numpy as np
import os
from laion_clap.training.params import parse_args






if __name__ == '__main__':
    args = parse_args()

    model_path = args.pretrained

    clotho_test_preprocessed_dir = "/fsx/yusong/clotho_test_set/test"

    cudnn.benchmark = True
    cudnn.deterministic = False

    audio_features_ensemble_all = []
    text_features_ensemble_all = []
    audio_features_mlp_ensemble_all = []
    text_features_mlp_ensemble_all = []
    logit_scale_a_ensemble_all = []
    logit_scale_t_ensemble_all = []


    device = torch.device('cuda')
    model, clap_model_cfg = create_model(
        args.amodel,
        args.tmodel,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        openai_model_cache_dir=os.path.expanduser(args.openai_model_cache_dir),
        skip_params=False
    )

    # load model
    checkpoint = torch.load(model_path, map_location=device)
    if "epoch" in checkpoint:
        # resuming a train checkpoint w/ epoch and optimizer state
        start_epoch = checkpoint["epoch"]
        sd = checkpoint["state_dict"]
        if next(iter(sd.items()))[0].startswith(
                "module"
        ):
            sd = {k[len("module."):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
    else:
        # loading a bare (model only) checkpoint for fine-tune or evaluation
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # take every 5th file because clotho has 5 texts for 1 audio
    test_file_list = sorted(glob.glob(f"{clotho_test_preprocessed_dir}/*.flac"))

    audio_features_all = []
    text_features_all = []
    audio_features_mlp_all = []
    text_features_mlp_all = []
    logit_scale_a_all = []
    logit_scale_t_all = []

    with torch.no_grad():
        for file_path in tqdm(test_file_list):
            json_path = file_path.replace(".flac", ".json")
            with open(json_path, "r") as f:
                json_data = json.load(f)
            audio, sr = librosa.load(file_path, sr=48000, mono=True)
            audio = torch.from_numpy(audio).to(device)
            audio = {'waveform': audio.unsqueeze(0), 'sample_rate': sr}
            text = json_data["text"]

            if args.tmodel == "transformer":
                from open_clip import tokenize
                text = tokenize(text)
            else:
                from laion_clap.training.data import tokenizer
                text = tokenizer(text, tmodel=args.tmodel)  # 5 texts for each audio

            audio_features, text_features, audio_features_mlp, text_features_mlp, logit_scale_a, logit_scale_t = \
                get_output_from_single_audio(audio, text, model, device)

            audio_features_all.append(audio_features.detach().cpu())
            text_features_all.append(text_features.detach().cpu())
            audio_features_mlp_all.append(audio_features_mlp.detach().cpu())
            text_features_mlp_all.append(text_features_mlp.detach().cpu())
            logit_scale_a_all.append(logit_scale_a.detach().cpu())
            logit_scale_t_all.append(logit_scale_t.detach().cpu())

    audio_features = torch.cat(audio_features_all)
    text_features = torch.cat(text_features_all)
    logit_scale_a = logit_scale_a_all[0]

    logits_per_audio = (logit_scale_a * audio_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_audio.t().detach().cpu()

    metrics = get_metrics(
        logits_per_text
    )

    print(metrics)