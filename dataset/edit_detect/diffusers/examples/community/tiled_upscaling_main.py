def main():
    model_id = 'stabilityai/stable-diffusion-x4-upscaler'
    pipe = StableDiffusionTiledUpscalePipeline.from_pretrained(model_id,
        revision='fp16', torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    image = Image.open('../../docs/source/imgs/diffusers_library.jpg')

    def callback(obj):
        print(f"progress: {obj['progress']:.4f}")
        obj['image'].save('diffusers_library_progress.jpg')
    final_image = pipe(image=image, prompt=
        'Black font, white background, vector', noise_level=40, callback=
        callback)
    final_image.save('diffusers_library.jpg')
