def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard',
        'thop'))
    run(**vars(opt))
