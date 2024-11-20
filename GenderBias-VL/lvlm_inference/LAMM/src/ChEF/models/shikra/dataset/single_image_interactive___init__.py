def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.image: Optional[Image.Image] = None
    self.roles = 'human', 'gpt'
    self.boxes = []
    self.points = []
    self.raw_conv = []
    self.conversations = []
