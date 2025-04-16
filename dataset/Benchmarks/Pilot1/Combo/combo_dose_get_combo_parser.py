def get_combo_parser():
    description = (
        'Build neural network based models to predict tumor response to drug pairs.'
        )
    parser = argparse.ArgumentParser(prog='combo_baseline', formatter_class
        =argparse.ArgumentDefaultsHelpFormatter, description=description)
    return combo.common_parser(parser)
