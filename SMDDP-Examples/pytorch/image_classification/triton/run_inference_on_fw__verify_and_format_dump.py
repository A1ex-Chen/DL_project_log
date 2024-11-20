def _verify_and_format_dump(args, ids, x, y_pred, y_real):
    data = {'outputs': y_pred, 'ids': {'ids': ids}}
    if args.dump_inputs:
        data['inputs'] = x
    if args.dump_labels:
        if not y_real:
            raise ValueError(
                'Found empty label values. Please provide labels in dataloader_fn or do not use --dump-labels argument'
                )
        data['labels'] = y_real
    return data
