def actual_main(args):
    from deepview_profile.analysis.session import AnalysisSession
    from deepview_profile.exceptions import AnalysisError
    if os.path.exists(args.output):
        print('ERROR: The specified output file already exists.', file=sys.
            stderr)
        sys.exit(1)
    try:
        project_root = os.getcwd()
        data = {'analysisState': {'message_type': 'analysis',
            'project_root': project_root, 'project_entry_point': args.
            entry_point, 'hardware_info': {}, 'throughput': {}, 'breakdown':
            {}, 'habitat': {}, 'additionalProviders': '', 'energy': {},
            'utilization': {}, 'ddp': {}}, 'epochs': 50, 'iterPerEpoch': 
            1000, 'encodedFiles': []}
        session = AnalysisSession.new_from(project_root, args.entry_point)
        release_memory()
        is_return_all = args.all
        with NVML() as nvml:
            data['analysisState']['hardware_info'] = hardware_information(nvml)
            if args.measure_breakdown or is_return_all:
                data['analysisState']['breakdown'] = next_message_to_dict(
                    measure_breakdown(session, nvml))
                operation_tree = data['analysisState']['breakdown'][
                    'operationTree']
                if not args.exclude_source and operation_tree is not None:
                    data['encodedFiles'] = files_encoded_unique(operation_tree)
        if args.measure_throughput or is_return_all:
            data['analysisState']['throughput'] = next_message_to_dict(
                measure_throughput(session))
        if args.habitat_predict or is_return_all:
            data['analysisState']['habitat'] = next_message_to_dict(
                habitat_predict(session))
        if args.measure_utilization or is_return_all:
            data['analysisState']['utilization'] = next_message_to_dict(
                measure_utilization(session))
        if args.energy_compute or is_return_all:
            data['analysisState']['energy'] = next_message_to_dict(
                energy_compute(session))
        if args.include_ddp:
            data['analysisState']['ddp'] = next_message_to_dict(ddp_analysis
                (session))
        with open(args.output, 'w') as json_file:
            json.dump(data, json_file, indent=4)
    except AnalysisError as ex:
        print_analysis_error(ex)
        sys.exit(1)
