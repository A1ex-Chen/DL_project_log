def __init__(self) ->None:
    super().__init__()
    self.embed_image = ImageEmbedding()
    self.embed_time = TimestepEmbedding_()
    down_0 = nn.ModuleList([ConvResblock(320, 320), ConvResblock(320, 320),
        ConvResblock(320, 320), Downsample(320)])
    down_1 = nn.ModuleList([ConvResblock(320, 640), ConvResblock(640, 640),
        ConvResblock(640, 640), Downsample(640)])
    down_2 = nn.ModuleList([ConvResblock(640, 1024), ConvResblock(1024, 
        1024), ConvResblock(1024, 1024), Downsample(1024)])
    down_3 = nn.ModuleList([ConvResblock(1024, 1024), ConvResblock(1024, 
        1024), ConvResblock(1024, 1024)])
    self.down = nn.ModuleList([down_0, down_1, down_2, down_3])
    self.mid = nn.ModuleList([ConvResblock(1024, 1024), ConvResblock(1024, 
        1024)])
    up_3 = nn.ModuleList([ConvResblock(1024 * 2, 1024), ConvResblock(1024 *
        2, 1024), ConvResblock(1024 * 2, 1024), ConvResblock(1024 * 2, 1024
        ), Upsample(1024)])
    up_2 = nn.ModuleList([ConvResblock(1024 * 2, 1024), ConvResblock(1024 *
        2, 1024), ConvResblock(1024 * 2, 1024), ConvResblock(1024 + 640, 
        1024), Upsample(1024)])
    up_1 = nn.ModuleList([ConvResblock(1024 + 640, 640), ConvResblock(640 *
        2, 640), ConvResblock(640 * 2, 640), ConvResblock(320 + 640, 640),
        Upsample(640)])
    up_0 = nn.ModuleList([ConvResblock(320 + 640, 320), ConvResblock(320 * 
        2, 320), ConvResblock(320 * 2, 320), ConvResblock(320 * 2, 320)])
    self.up = nn.ModuleList([up_0, up_1, up_2, up_3])
    self.output = ImageUnembedding()
