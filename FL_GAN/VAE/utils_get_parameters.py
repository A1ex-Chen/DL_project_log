def get_parameters(self):
    return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
