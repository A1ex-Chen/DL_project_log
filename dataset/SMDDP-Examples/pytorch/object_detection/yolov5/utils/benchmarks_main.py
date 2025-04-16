def main(opt):
    test(**vars(opt)) if opt.test else run(**vars(opt))
