def get_perfetto_object(filepath: str) ->TraceProcessor:
    """
    Input: Pytorch profiler trace
    Output: Handler to run SQL-like queries on trace. 
    """
    with open(filepath, 'rb') as f:
        raw_slices = orjson.loads(f.read())
    if isinstance(raw_slices, dict) and 'traceEvents' in raw_slices:
        raw_slices = raw_slices.pop('traceEvents', None)
    _convert_ids_int_string(raw_slices)
    _convert_negative_tids_to_positive(raw_slices)
    _remove_args(raw_slices)
    slices_bytes = orjson.dumps(raw_slices)
    slices_bytes = io.BytesIO(slices_bytes)
    try:
        tp = TraceProcessor(slices_bytes)
    except ConnectionResetError as e:
        tp = TraceProcessor(slices_bytes)
    interesting_fields = (
        'SELECT ts, dur, track_id, category, name, depth, cat, slice_id, id, arg_set_id FROM slice'
        )

    def query_dict(query):
        query = query.lower().replace('select * from slice', interesting_fields
            )
        try:
            query_iterator = tp.query(query)
        except Exception as e:
            print('[ERROR] Unable to run query: %s' % query)
            print(traceback.format_exc())
            print('\n\n')
            raise e
        return [item.__dict__ for item in query_iterator]
    tp.query_dict = query_dict
    return tp
