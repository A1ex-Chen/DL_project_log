def load_all_items(columns=[]):
    if len(columns) == 0:
        columns = ['item_id', 'title', 'price', 'pic_file', 'pic_url',
            'type', 'sales']
    item_data = Item.objects.all().values(*columns)
    items = pd.DataFrame.from_records(item_data, columns=columns)
    return items
