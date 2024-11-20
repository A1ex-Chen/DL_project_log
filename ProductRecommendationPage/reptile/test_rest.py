import numpy as np
import requests
import re
import pandas as pd
import time
import os
import tqdm
import urllib
from lxml import etree
import pymongo
from concurrent import futures


#UA列表
user_agent_list = [
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
            "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
            "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
            "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
			"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36",
			'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
        ]
len_userA = len(user_agent_list)

#构建代理IP池
with open("./verified_proxies.json", "r+") as fb:
	proxy_list = fb.read()
proxy_list = proxy_list.split('\n')
proxy_list = [eval(i) for i in proxy_list if len(i) != 0]
proxy_list = [{i['type'] : i['type'] + '://' + i['host'] + ':' + str(i['port'])} for i in proxy_list]
https_proxy_list = [i for i in proxy_list if 'https' in i.keys()]
len_https_proxy_list = len(https_proxy_list)
http_proxy_list = [i for i in proxy_list if 'http' in i.keys()]
len_http_proxy_list = len(https_proxy_list)
print(https_proxy_list[:5])


#淘宝
index_global = 1





	# printGoodsList(infoList)


###############################################################################

#京东
class CrawlDog:


        # self.mongo_collection.update_one(
        #     # 定位至相应数据
        #     {'item_id': params['item_id']},
        #     {
        #         '$set': {'comments_count': comments_count},  # 添加comments_count字段
        #         '$addToSet': {'comments': {'$each': comments}}  # 将comments中的每一项添加至comments字段中
        #     }, True)



if __name__ == '__main__':
    # 测试, 只爬取两页搜索页与两页评论
    # test = CrawlDog('耳机')
    # test.main(2, 2)
	main()



# import requests
# import re
#
#
# # def getHTMLText(url):
# #     try:
# #         r = requests.get(url, timeout=30)
# #         r.raise_for_status()
# #         r.encoding = r.apparent_encoding
# #         return r.text
# #     except:
# #         return ""
# #
# #
# # def parsePage(ilt, html):
# #     try:
# #         plt = re.findall(r'\"view_price\"\:\"[\d\.]*\"', html)
# #         tlt = re.findall(r'\"raw_title\"\:\".*?\"', html)
# #         for i in range(len(plt)):
# #             price = eval(plt[i].split(':')[1])
# #             title = eval(tlt[i].split(':')[1])
# #             ilt.append([price, title])
# #     except:
# #         print()
# #
# #
# # def printGoodsList(ilt):
# #     tplt = "{:4}\t{:8}\t{:16}"
# #     print(tplt.format("序号", "价格", "商品名称"))
# #     count = 0
# #     for t in ilt:
# #         count = count + 1
# #         print(tplt.format(count, t[0], t[1]))
# #
# #
# # def main():
# #     goods = '高达'
# #     depth = 3
# #     start_url = 'https://s.taobao.com/search?q=' + goods
# #     infoList = []
# #     for i in range(depth):
# #         try:
# #             url = start_url + '&s=' + str(44 * i)
# #             html = getHTMLText(url)
# #             parsePage(infoList, html)
# #         except:
# #             continue
# #     printGoodsList(infoList)
# #
# #
# # main()
#
# def get_html(url):
#     """获取源码html"""
#     try:
#         r = requests.get(url=url, timeout=10)
#         r.encoding = r.apparent_encoding
#         return r.text
#     except:
#         print("获取失败")
#
#
# def get_data(html, goodlist):
#     """使用re库解析商品名称和价格
#     tlist:商品名称列表
#     plist:商品价格列表"""
#     tlist = re.findall(r'\"raw_title\"\:\".*?\"', html)
#     plist = re.findall(r'\"view_price\"\:\"[\d\.]*\"', html)
#     for i in range(len(tlist)):
#         title = eval(tlist[i].split(':')[1])  # eval()函数简单说就是用于去掉字符串的引号
#         price = eval(plist[i].split(':')[1])
#         goodlist.append([title, price])
#
#
# def write_data(list, num):
#     # with open('E:/Crawler/case/taob2.txt', 'a') as data:
#     #    print(list, file=data)
#     for i in range(num):  # num控制把爬取到的商品写进多少到文本中
#         u = list[i]
#         with open('E:/taob.txt', 'a') as data:
#             print(u, file=data)
#
#
# def main():
#     goods = '水杯'
#     depth = 3   # 定义爬取深度，即翻页处理
#     start_url = 'https://s.taobao.com/search?q=python&commend=all&ssid=s5-e&search_type=mall&sourceId=tb.index&area=c2c&spm=a1z02.1.6856637.d4910789&bcoffset=0&ntoffset=6&p4ppushleft=1%2C48&s=88'
#     infoList = []
#     for i in range(depth):
#         try:
#             url = start_url + '&s=' + str(44 * i)  # 因为淘宝显示每页44个商品，第一页i=0,一次递增
#             html = get_html(url)
#             print(html)
#             get_data(html, infoList)
#         except:
#             continue
#     write_data(infoList, len(infoList))
#
#
# if __name__ == '__main__':
#     main()