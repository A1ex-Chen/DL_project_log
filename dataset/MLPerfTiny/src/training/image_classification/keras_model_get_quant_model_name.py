def get_quant_model_name():
    if os.path.exists('trained_models/trainedResnet.h5'):
        return 'trainedResnet'
    else:
        return 'pretrainedResnet'
