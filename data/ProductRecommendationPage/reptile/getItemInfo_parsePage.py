def parsePage(ilt, html, itemtype):
    global index_global
    try:
        plt = re.findall('\\"view_price\\"\\:\\"[\\d\\.]*\\"', html)
        tlt = re.findall('\\"raw_title\\"\\:\\".*?\\"', html)
        purl = re.findall('\\"pic_url\\"\\:\\".*?\\"', html)
        detail_url = re.findall('\\"detail_url\\"\\:\\".*?\\"', html)
        sales = re.findall('\\"view_sales\\"\\:\\".*?\\"', html)
        try:
            os.makedirs(f'../data/{itemtype}')
        except:
            pass
        for i in range(len(plt)):
            price = eval(plt[i].split(':')[1])
            title = eval(tlt[i].split(':')[1])
            pic_url = 'http:' + eval(purl[i].split(':')[1])
            sale = eval(sales[i].split(':')[1])
            sale = re.findall('\\d+', sale)
            sale = int(''.join(sale))
            try:
                iid = str(file_dict[itemtype]) + '_' + str(index_global)
                file = f'data/{itemtype}/{itemtype}_{index_global}.jpg'
                item = Item(item_id=iid, title=title, price=float(price),
                    pic_url=pic_url, pic_file=file, sales=sale, type=itemtype)
                item.save()
                print([iid, price, title, pic_url, itemtype, sale, file])
                urllib.request.urlretrieve(pic_url,
                    f'../data/{itemtype}/{itemtype}_{index_global}.jpg')
                img = Image.open(
                    f'../data/{itemtype}/{itemtype}_{index_global}.jpg')
                out = img.resize((200, 200))
                out.save(
                    f'../static/data/{itemtype}/{itemtype}_{index_global}.jpg',
                    'jpeg')
                ilt.append([index_global, price, title, pic_url, itemtype,
                    sales])
                index_global += 1
                time.sleep(np.random.choice(5, 1))
            except:
                print(f'{title}数据保存失败')
    except:
        return ''
