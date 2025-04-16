def index(request):
    item_column = ['item_id', 'title', 'price', 'pic_file', 'sales']
    try:
        uid = request.session['user_name']
        a = User.objects.filter(name__exact=uid).values(*['id'])
        a = pd.DataFrame.from_records(a, columns=['id']).values[0][0]
        res = Rec_Items.objects.filter(user_id__exact=str(a)).values(*[
            'rec_item'])
        res = pd.DataFrame.from_records(res, columns=['rec_item']).values[0][0]
        res = eval(res)
        res_list = [i for i, j in res]
        res_df = Item.objects.filter(item_id__in=res_list).values(*item_column)
        res_df = pd.DataFrame.from_records(res_df, columns=item_column)
        res_df = res_df.to_json(orient='records')
        res_df = eval(res_df)
        for j in res_df:
            j['pic_file'] = str(j['pic_file']).replace('\\', '')
    except:
        res_df = Item.objects.all().values(*item_column)
        res_df = pd.DataFrame.from_records(res_df, columns=item_column)
        res_df = res_df.sort_values(by='sales')[-40:]
        res_df = res_df.to_json(orient='records')
        res_df = eval(res_df)
        for j in res_df:
            j['pic_file'] = str(j['pic_file']).replace('\\', '')
    print(res_df)
    item_type = Item.objects.values_list('type')
    item_type = list(set(item_type))
    item_type_dict = {}
    for i in range(1, len(item_type) + 1):
        item_type_dict['#tab' + str(i)] = item_type[i - 1][0]
    it_dict = {}
    for i in range(1, len(item_type) + 1):
        a = Item.objects.filter(type__contains=item_type[i - 1][0]).values(*
            item_column)
        a = pd.DataFrame.from_records(a, columns=item_column)
        a = a.iloc[:10, :]
        a = a.to_json(orient='records')
        a = eval(a)
        for j in a:
            j['pic_file'] = str(j['pic_file']).replace('\\', '')
        it_dict['tab' + str(i)] = a
    return render(request, 'login/index.html', {'item_type_dict':
        item_type_dict, 'it_dict': it_dict, 'res_df': res_df})
