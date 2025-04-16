@classmethod
def setUpClass(cls):
    raw_image = load_image(
        'https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/pix2pix/cat_6.png'
        )
    raw_image = raw_image.convert('RGB').resize((512, 512))
    cls.raw_image = raw_image
