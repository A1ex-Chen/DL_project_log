def __init__(self, cid, model, trainloader: DataLoader, testloader:
    DataLoader, sample_rate: float, device: str) ->None:
    super().__init__()
    self.cid = cid
    self.model = model
    self.trainloader = trainloader
    self.tesloader = testloader
    self.device = device
    self.privacy_engine = PrivacyEngine()
