def combine_partial_results(partial_results) ->List:
    """Concatenate partial results into one file, then sort it by id."""
    records = []
    for partial_result in partial_results:
        records.extend(partial_result)
    records = list(sorted(records, key=lambda x: x['id']))
    preds = [x['pred'] for x in records]
    return preds
