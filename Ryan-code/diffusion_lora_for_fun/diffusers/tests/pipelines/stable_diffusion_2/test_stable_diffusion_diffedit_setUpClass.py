@classmethod
def setUpClass(cls):
    raw_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/diffedit/fruit.png'
        )
    raw_image = raw_image.convert('RGB').resize((768, 768))
    cls.raw_image = raw_image
