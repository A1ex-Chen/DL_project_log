"""
This script ports models from VQ-diffusion (https://github.com/microsoft/VQ-Diffusion) to diffusers.

It currently only supports porting the ITHQ dataset.

ITHQ dataset:
```sh
# From the root directory of diffusers.

# Download the VQVAE checkpoint
$ wget https://facevcstandard.blob.core.windows.net/v-zhictang/Improved-VQ-Diffusion_model_release/ithq_vqvae.pth?sv=2020-10-02&st=2022-05-30T15%3A17%3A18Z&se=2030-05-31T15%3A17%3A00Z&sr=b&sp=r&sig=1jVavHFPpUjDs%2FTO1V3PTezaNbPp2Nx8MxiWI7y6fEY%3D -O ithq_vqvae.pth

# Download the VQVAE config
# NOTE that in VQ-diffusion the documented file is `configs/ithq.yaml` but the target class
# `image_synthesis.modeling.codecs.image_codec.ema_vqvae.PatchVQVAE`
# loads `OUTPUT/pretrained_model/taming_dvae/config.yaml`
$ wget https://raw.githubusercontent.com/microsoft/VQ-Diffusion/main/OUTPUT/pretrained_model/taming_dvae/config.yaml -O ithq_vqvae.yaml

# Download the main model checkpoint
$ wget https://facevcstandard.blob.core.windows.net/v-zhictang/Improved-VQ-Diffusion_model_release/ithq_learnable.pth?sv=2020-10-02&st=2022-05-30T10%3A22%3A06Z&se=2030-05-31T10%3A22%3A00Z&sr=b&sp=r&sig=GOE%2Bza02%2FPnGxYVOOPtwrTR4RA3%2F5NVgMxdW4kjaEZ8%3D -O ithq_learnable.pth

# Download the main model config
$ wget https://raw.githubusercontent.com/microsoft/VQ-Diffusion/main/configs/ithq.yaml -O ithq.yaml

# run the convert script
$ python ./scripts/convert_vq_diffusion_to_diffusers.py \
    --checkpoint_path ./ithq_learnable.pth \
    --original_config_file ./ithq.yaml \
    --vqvae_checkpoint_path ./ithq_vqvae.pth \
    --vqvae_original_config_file ./ithq_vqvae.yaml \
    --dump_path <path to save pre-trained `VQDiffusionPipeline`>
```
"""

import argparse
import tempfile

import torch
import yaml
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import CLIPTextModel, CLIPTokenizer
from yaml.loader import FullLoader

from diffusers import Transformer2DModel, VQDiffusionPipeline, VQDiffusionScheduler, VQModel
from diffusers.pipelines.vq_diffusion.pipeline_vq_diffusion import LearnedClassifierFreeSamplingEmbeddings


# vqvae model

PORTED_VQVAES = ["image_synthesis.modeling.codecs.image_codec.patch_vqgan.PatchVQGAN"]












# done vqvae model

# vqvae checkpoint












# done vqvae checkpoint

# transformer model

PORTED_DIFFUSIONS = ["image_synthesis.modeling.transformers.diffusion_transformer.DiffusionTransformer"]
PORTED_TRANSFORMERS = ["image_synthesis.modeling.transformers.transformer_utils.Text2ImageTransformer"]
PORTED_CONTENT_EMBEDDINGS = ["image_synthesis.modeling.embeddings.dalle_mask_image_embedding.DalleMaskImageEmbedding"]




# done transformer model

# transformer checkpoint










# done transformer checkpoint




# We take separate arguments for the vqvae because the ITHQ vqvae config file
# is separate from the config file for the rest of the model.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vqvae_checkpoint_path",
        default=None,
        type=str,
        required=True,
        help="Path to the vqvae checkpoint to convert.",
    )

    parser.add_argument(
        "--vqvae_original_config_file",
        default=None,
        type=str,
        required=True,
        help="The YAML config file corresponding to the original architecture for the vqvae.",
    )

    parser.add_argument(
        "--checkpoint_path", default=None, type=str, required=True, help="Path to the checkpoint to convert."
    )

    parser.add_argument(
        "--original_config_file",
        default=None,
        type=str,
        required=True,
        help="The YAML config file corresponding to the original architecture.",
    )

    parser.add_argument("--dump_path", default=None, type=str, required=True, help="Path to the output model.")

    parser.add_argument(
        "--checkpoint_load_device",
        default="cpu",
        type=str,
        required=False,
        help="The device passed to `map_location` when loading checkpoints.",
    )

    # See link for how ema weights are always selected
    # https://github.com/microsoft/VQ-Diffusion/blob/3c98e77f721db7c787b76304fa2c96a36c7b00af/inference_VQ_Diffusion.py#L65
    parser.add_argument(
        "--no_use_ema",
        action="store_true",
        required=False,
        help=(
            "Set to not use the ema weights from the original VQ-Diffusion checkpoint. You probably do not want to set"
            " it as the original VQ-Diffusion always uses the ema weights when loading models."
        ),
    )

    args = parser.parse_args()

    use_ema = not args.no_use_ema

    print(f"loading checkpoints to {args.checkpoint_load_device}")

    checkpoint_map_location = torch.device(args.checkpoint_load_device)

    # vqvae_model

    print(f"loading vqvae, config: {args.vqvae_original_config_file}, checkpoint: {args.vqvae_checkpoint_path}")

    vqvae_original_config = read_config_file(args.vqvae_original_config_file).model
    vqvae_checkpoint = torch.load(args.vqvae_checkpoint_path, map_location=checkpoint_map_location)["model"]

    with init_empty_weights():
        vqvae_model = vqvae_model_from_original_config(vqvae_original_config)

    vqvae_diffusers_checkpoint = vqvae_original_checkpoint_to_diffusers_checkpoint(vqvae_model, vqvae_checkpoint)

    with tempfile.NamedTemporaryFile() as vqvae_diffusers_checkpoint_file:
        torch.save(vqvae_diffusers_checkpoint, vqvae_diffusers_checkpoint_file.name)
        del vqvae_diffusers_checkpoint
        del vqvae_checkpoint
        load_checkpoint_and_dispatch(vqvae_model, vqvae_diffusers_checkpoint_file.name, device_map="auto")

    print("done loading vqvae")

    # done vqvae_model

    # transformer_model

    print(
        f"loading transformer, config: {args.original_config_file}, checkpoint: {args.checkpoint_path}, use ema:"
        f" {use_ema}"
    )

    original_config = read_config_file(args.original_config_file).model

    diffusion_config = original_config["params"]["diffusion_config"]
    transformer_config = original_config["params"]["diffusion_config"]["params"]["transformer_config"]
    content_embedding_config = original_config["params"]["diffusion_config"]["params"]["content_emb_config"]

    pre_checkpoint = torch.load(args.checkpoint_path, map_location=checkpoint_map_location)

    if use_ema:
        if "ema" in pre_checkpoint:
            checkpoint = {}
            for k, v in pre_checkpoint["model"].items():
                checkpoint[k] = v

            for k, v in pre_checkpoint["ema"].items():
                # The ema weights are only used on the transformer. To mimic their key as if they came
                # from the state_dict for the top level model, we prefix with an additional "transformer."
                # See the source linked in the args.use_ema config for more information.
                checkpoint[f"transformer.{k}"] = v
        else:
            print("attempted to load ema weights but no ema weights are specified in the loaded checkpoint.")
            checkpoint = pre_checkpoint["model"]
    else:
        checkpoint = pre_checkpoint["model"]

    del pre_checkpoint

    with init_empty_weights():
        transformer_model = transformer_model_from_original_config(
            diffusion_config, transformer_config, content_embedding_config
        )

    diffusers_transformer_checkpoint = transformer_original_checkpoint_to_diffusers_checkpoint(
        transformer_model, checkpoint
    )

    # classifier free sampling embeddings interlude

    # The learned embeddings are stored on the transformer in the original VQ-diffusion. We store them on a separate
    # model, so we pull them off the checkpoint before the checkpoint is deleted.

    learnable_classifier_free_sampling_embeddings = diffusion_config["params"].learnable_cf

    if learnable_classifier_free_sampling_embeddings:
        learned_classifier_free_sampling_embeddings_embeddings = checkpoint["transformer.empty_text_embed"]
    else:
        learned_classifier_free_sampling_embeddings_embeddings = None

    # done classifier free sampling embeddings interlude

    with tempfile.NamedTemporaryFile() as diffusers_transformer_checkpoint_file:
        torch.save(diffusers_transformer_checkpoint, diffusers_transformer_checkpoint_file.name)
        del diffusers_transformer_checkpoint
        del checkpoint
        load_checkpoint_and_dispatch(transformer_model, diffusers_transformer_checkpoint_file.name, device_map="auto")

    print("done loading transformer")

    # done transformer_model

    # text encoder

    print("loading CLIP text encoder")

    clip_name = "openai/clip-vit-base-patch32"

    # The original VQ-Diffusion specifies the pad value by the int used in the
    # returned tokens. Each model uses `0` as the pad value. The transformers clip api
    # specifies the pad value via the token before it has been tokenized. The `!` pad
    # token is the same as padding with the `0` pad value.
    pad_token = "!"

    tokenizer_model = CLIPTokenizer.from_pretrained(clip_name, pad_token=pad_token, device_map="auto")

    assert tokenizer_model.convert_tokens_to_ids(pad_token) == 0

    text_encoder_model = CLIPTextModel.from_pretrained(
        clip_name,
        # `CLIPTextModel` does not support device_map="auto"
        # device_map="auto"
    )

    print("done loading CLIP text encoder")

    # done text encoder

    # scheduler

    scheduler_model = VQDiffusionScheduler(
        # the scheduler has the same number of embeddings as the transformer
        num_vec_classes=transformer_model.num_vector_embeds
    )

    # done scheduler

    # learned classifier free sampling embeddings

    with init_empty_weights():
        learned_classifier_free_sampling_embeddings_model = LearnedClassifierFreeSamplingEmbeddings(
            learnable_classifier_free_sampling_embeddings,
            hidden_size=text_encoder_model.config.hidden_size,
            length=tokenizer_model.model_max_length,
        )

    learned_classifier_free_sampling_checkpoint = {
        "embeddings": learned_classifier_free_sampling_embeddings_embeddings.float()
    }

    with tempfile.NamedTemporaryFile() as learned_classifier_free_sampling_checkpoint_file:
        torch.save(learned_classifier_free_sampling_checkpoint, learned_classifier_free_sampling_checkpoint_file.name)
        del learned_classifier_free_sampling_checkpoint
        del learned_classifier_free_sampling_embeddings_embeddings
        load_checkpoint_and_dispatch(
            learned_classifier_free_sampling_embeddings_model,
            learned_classifier_free_sampling_checkpoint_file.name,
            device_map="auto",
        )

    # done learned classifier free sampling embeddings

    print(f"saving VQ diffusion model, path: {args.dump_path}")

    pipe = VQDiffusionPipeline(
        vqvae=vqvae_model,
        transformer=transformer_model,
        tokenizer=tokenizer_model,
        text_encoder=text_encoder_model,
        learned_classifier_free_sampling_embeddings=learned_classifier_free_sampling_embeddings_model,
        scheduler=scheduler_model,
    )
    pipe.save_pretrained(args.dump_path)

    print("done writing VQ diffusion model")