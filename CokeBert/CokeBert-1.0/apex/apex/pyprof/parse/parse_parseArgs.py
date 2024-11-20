def parseArgs():
    parser = argparse.ArgumentParser(prog=sys.argv[0], description=
        'Parse SQL (nvvp) db.')
    parser.add_argument('file', type=str, default=None, help=
        'SQL db (nvvp) file.')
    args = parser.parse_args()
    return args
