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
            writer.writerow(['batch_size', 'run_time_ms_slope',
                'run_time_ms_bias', 'memory_usage_bytes_slope',
                'memory_usage_bytes_bias'])
            project_root = os.getcwd()
            for batch_size in args.batch_sizes:
                session = AnalysisSession.new_from(project_root, args.
                    entry_point)
                memory_model, run_time_model = get_model(session, batch_size)
                writer.writerow([batch_size, *run_time_model, *memory_model])
    except AnalysisError as ex:
        print_analysis_error(ex)
        sys.exit(1)
