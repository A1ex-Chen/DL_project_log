def detail(request, item_id):
    item_column = ['item_id', 'title', 'price', 'pic_file', 'sales']
    print(item_id)
    item_detail = Item.objects.filter(item_id__exact=item_id).values(*
        item_column)
    item_detail = pd.DataFrame.from_records(item_detail, columns=item_column)
    item_detail = item_detail.to_json(orient='records')
    item_detail = eval(item_detail)[0]
    item_detail['pic_file'] = str(item_detail['pic_file']).replace('\\', '')
    print(item_detail)
    return render(request, 'login/detail.html', {'item_detail': item_detail})
