def load_headers(desc_headers, train_headers, header_url):
    desc_headers = candle.get_file(desc_headers, header_url + desc_headers,
        cache_subdir='Pilot1')
    train_headers = candle.get_file(train_headers, header_url +
        train_headers, cache_subdir='Pilot1')
    with open(desc_headers) as f:
        reader = csv.reader(f, delimiter=',')
        dh_row = next(reader)
        dh_row = [x.strip() for x in dh_row]
    dh_dict = {}
    for i in range(len(dh_row)):
        dh_dict[dh_row[i]] = i
    with open(train_headers) as f:
        reader = csv.reader(f, delimiter=',')
        th_list = next(reader)
        th_list = [x.strip() for x in th_list]
    return dh_dict, th_list
