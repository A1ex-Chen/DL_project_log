def main():
    args = ArgumentParser()
    args.add_argument('--model_256', action='store_true')
    args.add_argument('--write_to', type=str, required=False, default=None)
    args.add_argument('--transformer_path', type=str, required=False,
        default=None)
    args = args.parse_args()
    transformer_path = args.transformer_path
    subfolder = 'transformer'
    if transformer_path is None:
        if args.model_256:
            transformer_path = 'openMUSE/muse-256'
        else:
            transformer_path = (
                '../research-run-512-checkpoints/research-run-512-with-downsample-checkpoint-554000/unwrapped_model/'
                )
            subfolder = None
    old_transformer = MaskGiTUViT.from_pretrained(transformer_path,
        subfolder=subfolder)
    old_transformer.to(device)
    old_vae = VQGANModel.from_pretrained('openMUSE/muse-512', subfolder='vae')
    old_vae.to(device)
    vqvae = make_vqvae(old_vae)
    tokenizer = CLIPTokenizer.from_pretrained('openMUSE/muse-512',
        subfolder='text_encoder')
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        'openMUSE/muse-512', subfolder='text_encoder')
    text_encoder.to(device)
    transformer = make_transformer(old_transformer, args.model_256)
    scheduler = AmusedScheduler(mask_token_id=old_transformer.config.
        mask_token_id)
    new_pipe = AmusedPipeline(vqvae=vqvae, tokenizer=tokenizer,
        text_encoder=text_encoder, transformer=transformer, scheduler=scheduler
        )
    old_pipe = OldPipelineMuse(vae=old_vae, transformer=old_transformer,
        text_encoder=text_encoder, tokenizer=tokenizer)
    old_pipe.to(device)
    if args.model_256:
        transformer_seq_len = 256
        orig_size = 256, 256
    else:
        transformer_seq_len = 1024
        orig_size = 512, 512
    old_out = old_pipe('dog', generator=torch.Generator(device).manual_seed
        (0), transformer_seq_len=transformer_seq_len, orig_size=orig_size,
        timesteps=12)[0]
    new_out = new_pipe('dog', generator=torch.Generator(device).manual_seed(0)
        ).images[0]
    old_out = np.array(old_out)
    new_out = np.array(new_out)
    diff = np.abs(old_out.astype(np.float64) - new_out.astype(np.float64))
    print('skipping pipeline full equivalence check')
    print(
        f'max diff: {diff.max()}, diff.sum() / diff.size {diff.sum() / diff.size}'
        )
    if args.model_256:
        assert diff.max() <= 3
        assert diff.sum() / diff.size < 0.7
    else:
        assert diff.max() <= 1
        assert diff.sum() / diff.size < 0.4
    if args.write_to is not None:
        new_pipe.save_pretrained(args.write_to)
