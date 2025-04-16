def _get_perfetto_object(self, filepath):
    with open(filepath, 'rb') as f:
        raw_slices = orjson.loads(f.read())
    self._filter_traces(raw_slices)
    if isinstance(raw_slices, dict) and 'traceEvents' in raw_slices:
        raw_slices = raw_slices.pop('traceEvents', None)
    self._convert_ids_int_string(raw_slices)
    self._convert_negative_tids_to_positive(raw_slices)
    self._remove_args(raw_slices)
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
