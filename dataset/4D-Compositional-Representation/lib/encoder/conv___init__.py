def __init__(self, c_dim=128, hidden_dim=32, **kwargs):
    """ Initialisation.

        Args:
            c_dim (int): output dimension of the latent embedding
        """
    super().__init__()
    self.conv0 = nn.Conv3d(3, hidden_dim, 3, stride=(1, 2, 2), padding=1)
    self.conv1 = nn.Conv3d(hidden_dim, hidden_dim * 2, 3, stride=(2, 2, 2),
        padding=1)
    self.conv2 = nn.Conv3d(hidden_dim * 2, hidden_dim * 4, 3, stride=(1, 2,
        2), padding=1)
    self.conv3 = nn.Conv3d(hidden_dim * 4, hidden_dim * 8, 3, stride=(2, 2,
        2), padding=1)
    self.conv4 = nn.Conv3d(hidden_dim * 8, hidden_dim * 16, 3, stride=(2, 2,
        2), padding=1)
    self.conv5 = nn.Conv3d(hidden_dim * 16, hidden_dim * 16, 3, stride=(2, 
        2, 2), padding=1)
    self.fc_out = nn.Linear(hidden_dim * 16, c_dim)
    self.actvn = nn.ReLU()
