def query_dict(query):
    query = query.lower().replace('select * from slice', interesting_fields)
    try:
        query_iterator = tp.query(query)
    except Exception as e:
        print('[ERROR] Unable to run query: %s' % query)
        print(traceback.format_exc())
        print('\n\n')
        raise e
    return [item.__dict__ for item in query_iterator]
