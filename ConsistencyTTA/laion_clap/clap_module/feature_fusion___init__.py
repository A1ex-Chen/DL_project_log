def __init__(self, channels=64, r=4, type='2D'):
    super(AFF, self).__init__()
    inter_channels = int(channels // r)
    if type == '1D':
        self.local_att = nn.Sequential(nn.Conv1d(channels, inter_channels,
            kernel_size=1, stride=1, padding=0), nn.BatchNorm1d(
            inter_channels), nn.ReLU(inplace=True), nn.Conv1d(
            inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels))
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(
            channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels), nn.ReLU(inplace=True), nn.
            Conv1d(inter_channels, channels, kernel_size=1, stride=1,
            padding=0), nn.BatchNorm1d(channels))
    elif type == '2D':
        self.local_att = nn.Sequential(nn.Conv2d(channels, inter_channels,
            kernel_size=1, stride=1, padding=0), nn.BatchNorm2d(
            inter_channels), nn.ReLU(inplace=True), nn.Conv2d(
            inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels))
        self.global_att = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(
            channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True), nn.
            Conv2d(inter_channels, channels, kernel_size=1, stride=1,
            padding=0), nn.BatchNorm2d(channels))
    else:
        raise f'the type is not supported.'
    self.sigmoid = nn.Sigmoid()
