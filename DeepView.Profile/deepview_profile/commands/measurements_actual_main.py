def actual_main(args):
    from deepview_profile.analysis.session import AnalysisSession
    from deepview_profile.exceptions import AnalysisError
    if os.path.exists(args.output):
        print('ERROR: The specified output file already exists.', file=sys.
            stderr)
        sys.exit(1)
    try:
        with open(args.output, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['batch_size', 'trial', 'samples_per_second',
                'memory_usage_bytes'])
            project_root = os.getcwd()
            for batch_size in args.batch_sizes:
                for trial in range(args.trials):
                    session = AnalysisSession.new_from(project_root, args.
                        entry_point)
                    samples_per_second, memory_usage_bytes = make_measurements(
                        session, batch_size)
                    writer.writerow([batch_size, trial, samples_per_second,
                        memory_usage_bytes])
    except AnalysisError as ex:
        print_analysis_error(ex)
        sys.exit(1)
