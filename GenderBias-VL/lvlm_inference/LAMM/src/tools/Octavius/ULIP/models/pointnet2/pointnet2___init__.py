def __init__(self, normal_channel=False):
    super(Pointnet2_Msg, self).__init__()
    in_channel = 3 if normal_channel else 0
    self.normal_channel = normal_channel
    self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128
        ], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
    self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128
        ], 320, [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
    self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512,
        1024], True)
    self.fc1 = nn.Linear(1024, 512)
    self.bn1 = nn.BatchNorm1d(512)
    self.drop1 = nn.Dropout(0.4)
    self.fc2 = nn.Linear(512, 256)
    self.bn2 = nn.BatchNorm1d(256)
    self.drop2 = nn.Dropout(0.5)
