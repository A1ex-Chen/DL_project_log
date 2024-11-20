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
        for i in tqdm.tqdm(range(len(plt))):
            price = eval(plt[i].split(':')[1])
            title = eval(tlt[i].split(':')[1])
            pic_url = 'http:' + eval(purl[i].split(':')[1])
            try:
                urllib.request.urlretrieve(pic_url,
                    f'../data/{itemtype}/{itemtype}_{index_global}.jpg')
                ilt.append([index_global, price, title, pic_url, itemtype, 0])
                index_global += 1
                time.sleep(np.random.choice(5, 1))
            except:
                print(f'{title}数据保存失败')
    except:
        return ''
