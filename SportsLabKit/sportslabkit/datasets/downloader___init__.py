def __init__(self) ->None:
    """An downloader to download the soccertrack dataset from Kaggle."""
    self.api = authenticate()
    self.dataset_owner = 'atomscott'
    self.dataset_name = 'soccertrack'
