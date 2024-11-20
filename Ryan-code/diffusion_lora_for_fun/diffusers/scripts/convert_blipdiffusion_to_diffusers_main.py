def main(args):
    model, _, _ = load_model_and_preprocess('blip_diffusion', 'base',
        device='cpu', is_eval=True)
    save_blip_diffusion_model(model.state_dict(), args)
