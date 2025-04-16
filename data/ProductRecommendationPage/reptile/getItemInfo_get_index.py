def get_index(self, page):
    """
        从搜索页获取相应信息并存入数据库
        :param page: 搜索页的页码
        :return: 商品的id
        """
    url = 'https://search.jd.com/Search?keyword=%s&enc=utf-8&page=%d' % (self
        .keyword, page)
    n = np.random.choice(len_userA, 1)[0]
    m = np.random.choice(len_https_proxy_list, 1)[0]
    index_headers = {'accept':
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3'
        , 'accept-encoding': 'gzip, deflate, br', 'Accept-Charset': 'utf-8',
        'accept-language':
        'zh,en-US;q=0.9,en;q=0.8,zh-TW;q=0.7,zh-CN;q=0.6', 'user-agent':
        user_agent_list[n]}
    rsp = requests.get(url=url, headers=index_headers, proxies=
        https_proxy_list[m]).content.decode()
    rsp = etree.HTML(rsp)
    items = rsp.xpath('//li[contains(@class, "gl-item")]')
    for item in items:
        try:
            info = dict()
            info['名称'] = ''.join(item.xpath(
                './/div[@class="p-name p-name-type-2"]//em//text()'))
            info['url'] = 'https:' + item.xpath(
                './/div[@class="p-name p-name-type-2"]/a/@href')[0]
            info['store'] = item.xpath('.//div[@class="p-shop"]/span/a/text()'
                )[0]
            info['store_url'] = 'https' + item.xpath(
                './/div[@class="p-shop"]/span/a/@href')[0]
            info['序号'] = info.get('url').split('/')[-1][:-5]
            info['价格'] = item.xpath('.//div[@class="p-price"]//i/text()')[0]
            info['图片URL'] = item.xpath(
                './/[@class="p-img"]//i/@source-data-lazy-img')[0]
            info['类型'] = self.keyword
            info['comments'] = []
            yield info['序号']
        except IndexError:
            print('item信息不全, drop!')
            continue
