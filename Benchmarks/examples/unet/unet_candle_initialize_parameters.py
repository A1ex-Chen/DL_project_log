def initialize_parameters():
    unet_common = unet.UNET(unet.file_path, 'unet_params.txt', 'keras',
        prog='unet_example', desc='UNET example')
    gParameters = candle.finalize_parameters(unet_common)
    return gParameters
