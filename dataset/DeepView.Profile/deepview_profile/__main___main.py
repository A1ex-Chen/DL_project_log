def main():
    parser = argparse.ArgumentParser(prog='DeepView', description=
        'DeepView: Interactive Neural Network Performance Profiler, Visualizer, and Debugger for PyTorch'
        )
    parser.add_argument('-v', '--version', action='store_true', help=
        'Print the version and exit.')
    subparsers = parser.add_subparsers(title='Commands')
    deepview_profile.commands.interactive.register_command(subparsers)
    deepview_profile.commands.memory.register_command(subparsers)
    deepview_profile.commands.time.register_command(subparsers)
    deepview_profile.commands.analysis.register_command(subparsers)
    args = parser.parse_args()
    if args.version:
        print('DeepView Command Line Interface', 'v' + deepview_profile.
            __version__)
        return
    if 'func' not in args:
        parser.print_help()
        sys.exit(1)
    args.func(args)
