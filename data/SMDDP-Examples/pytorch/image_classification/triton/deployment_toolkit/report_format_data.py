def format_data(data: List[Dict]) ->List[Dict]:
    formatted_data = list()
    for item in data:
        formatted_item = format_keys(data=item)
        formatted_data.append(formatted_item)
    return formatted_data
