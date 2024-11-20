import numpy as np
import requests
import re
import pandas as pd
import time
import os
import tqdm
import urllib

from PIL import Image
import django
from decimal import Decimal

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "RecSysInItem.settings")
django.setup()

from items.models import Item



# file_dict = {'乐器':1, '保健品':2, '办公':3, '化妆品':4, '图书':5, '学习':6,
#              '家具':7, '家电':8, '帽子':9, '手机':10, '杯子':11, '汽车':12,
#              '玩具':13, '珠宝':14, '生鲜':15, '电脑':16, '眼镜':17, '衣服':18, '零食':19}
#
#
# #UA列表
# user_agent_list = [
#             "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/22.0.1207.1 Safari/537.1",
#             "Mozilla/5.0 (X11; CrOS i686 2268.111.0) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.57 Safari/536.11",
#             "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1092.0 Safari/536.6",
#             "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.6 (KHTML, like Gecko) Chrome/20.0.1090.0 Safari/536.6",
#             "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/19.77.34.5 Safari/537.1",
#             "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.9 Safari/536.5",
#             "Mozilla/5.0 (Windows NT 6.0) AppleWebKit/536.5 (KHTML, like Gecko) Chrome/19.0.1084.36 Safari/536.5",
#             "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
#             "Mozilla/5.0 (Windows NT 5.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
#             "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_0) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1063.0 Safari/536.3",
#             "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
#             "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1062.0 Safari/536.3",
#             "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
#             "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
#             "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.1 Safari/536.3",
#             "Mozilla/5.0 (Windows NT 6.2) AppleWebKit/536.3 (KHTML, like Gecko) Chrome/19.0.1061.0 Safari/536.3",
#             "Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/535.24 (KHTML, like Gecko) Chrome/19.0.1055.1 Safari/535.24",
# 			"Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",
# 			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36",
# 			'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
#         ]
# len_userA = len(user_agent_list)
#
# #构建代理IP池
# with open("./verified_proxies.json", "r+") as fb:
# 	proxy_list = fb.read()
# proxy_list = proxy_list.split('\n')
# proxy_list = [eval(i) for i in proxy_list if len(i) != 0]
# proxy_list = [{i['type'] : i['type'] + '://' + i['host'] + ':' + str(i['port'])} for i in proxy_list]
# https_proxy_list = [i for i in proxy_list if 'https' in i.keys()]
# len_https_proxy_list = len(https_proxy_list)
# http_proxy_list = [i for i in proxy_list if 'http' in i.keys()]
# len_http_proxy_list = len(https_proxy_list)
# # print(https_proxy_list[:5])
#
#
# #淘宝
# index_global = 500






		# toCsv(infoList, goods[j])
	# printGoodsList(infoList)


###############################################################################


class CrawlDog:


        # self.mongo_collection.update_one(
        #     # 定位至相应数据
        #     {'item_id': params['item_id']},
        #     {
        #         '$set': {'comments_count': comments_count},  # 添加comments_count字段
        #         '$addToSet': {'comments': {'$each': comments}}  # 将comments中的每一项添加至comments字段中
        #     }, True)



######################################################################

if __name__ == '__main__':
    # 测试, 只爬取两页搜索页与两页评论
    # test = CrawlDog('耳机')
    # test.main(2, 2)

    file_dict = {'乐器': 1, '保健品': 2, '办公': 3, '化妆品': 4, '图书': 5, '学习': 6,
                 '家具': 7, '家电': 8, '帽子': 9, '手机': 10, '杯子': 11, '汽车': 12,
                 '玩具': 13, '珠宝': 14, '生鲜': 15, '电脑': 16, '眼镜': 17, '衣服': 18, '零食': 19}

    # UA列表
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

    # 构建代理IP池
    with open("verified_proxies.json", "r+") as fb:
	    proxy_list = fb.read()
    proxy_list = proxy_list.split('\n')
    proxy_list = [eval(i) for i in proxy_list if len(i) != 0]
    proxy_list = [{i['type']: i['type'] + '://' + i['host'] + ':' + str(i['port'])} for i in proxy_list]
    https_proxy_list = [i for i in proxy_list if 'https' in i.keys()]
    len_https_proxy_list = len(https_proxy_list)
    http_proxy_list = [i for i in proxy_list if 'http' in i.keys()]
    len_http_proxy_list = len(https_proxy_list)
    # print(https_proxy_list[:5])


    # 淘宝
    index_global = 500

    main()
