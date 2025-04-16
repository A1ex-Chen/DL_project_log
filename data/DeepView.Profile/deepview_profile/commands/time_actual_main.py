def actual_main(args):
    from deepview_profile.analysis.session import AnalysisSession
    from deepview_profile.exceptions import AnalysisError
    if os.path.exists(args.output):
        print('ERROR: The specified output file already exists.', file=sys.
            stderr)
        sys.exit(1)
    try:
        project_root = os.getcwd()
        session = AnalysisSession.new_from(project_root, args.entry_point)
        session.generate_run_time_breakdown_report(save_report_to=args.output)
    except AnalysisError as ex:
        print_analysis_error(ex)
        sys.exit(1)
