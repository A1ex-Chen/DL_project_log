def parseArgs():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=
        'Run popular imagenet models.')
    parser.add_argument('-m', type=str, default='resnet50', choices=[
        'alexnet', 'densenet121', 'densenet161', 'densenet169',
        'densenet201', 'googlenet', 'mnasnet0_5', 'mnasnet0_75',
        'mnasnet1_0', 'mnasnet1_3', 'mobilenet_v2', 'resnet18', 'resnet34',
        'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
        'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
        'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5',
        'shufflenet_v2_x2_0', 'squeezenet1_0', 'squeezenet1_1', 'vgg11',
        'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19',
        'vgg19_bn', 'inception_v3'], help='Model.')
    parser.add_argument('-b', type=int, default=32, help='Batch size.')
    parser.add_argument('-o', type=str, default='adam', choices=['adam',
        'sgd'], help='Optimizer.')
    args = parser.parse_args()
    return args
