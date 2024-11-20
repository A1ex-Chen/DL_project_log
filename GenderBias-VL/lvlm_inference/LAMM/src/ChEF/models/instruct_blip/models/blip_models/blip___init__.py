def __init__(self):
    super().__init__()
    transformers_version = version.parse(transformers.__version__)
    assert transformers_version < version.parse('4.27'
        ), 'BLIP models are not compatible with transformers>=4.27, run pip install transformers==4.25 to downgrade'
