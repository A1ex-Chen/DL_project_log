def main(opt):
    for opt.weights in (opt.weights if isinstance(opt.weights, list) else [
        opt.weights]):
        run(**vars(opt))
