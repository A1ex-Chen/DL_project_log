def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))
