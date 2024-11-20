def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        i = len(weights) - 1
        while len(weights) > 0:
            weights.pop()
            model = models[i]
            sub_dir = 'controlnet'
            model.save_pretrained(os.path.join(output_dir, sub_dir))
            i -= 1
