def parse():
    parser = argparse.ArgumentParser(description='Load model configuration')
    parser.add_argument('--config', help=
        'Configuration file to apply on top of base')
    parsed, _ = parser.parse_known_args()
    return parsed
