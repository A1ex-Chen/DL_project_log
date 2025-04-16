def command_line_chk():
    parser = argparse.ArgumentParser(description=
        'Without option argument, it will not run properly.')
    parser.add_argument('-v', '--version', action='store_true', help=
        'show application version')
    parser.add_argument('-e', '--eval', action='store_true', help=
        'run mode Evaluation')
    parser.add_argument('-d', '--dev', action='store_true', help=
        'run mode Development')
    args = parser.parse_args()
    if args.version:
        print('===============================')
        print('DCASE 2020 task 2 baseline\nversion {}'.format(__versions__))
        print('===============================\n')
    if args.eval ^ args.dev:
        if args.dev:
            flag = True
        else:
            flag = False
    else:
        flag = None
        print('incorrect argument')
        print("please set option argument '--dev' or '--eval'")
    return flag
