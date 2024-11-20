import copy
from bs4 import BeautifulSoup
from tqdm import tqdm
from . import ctrl_common
from . import reptile_base

# region 杂项
# 关键词
WORDS = '碧蓝航线'

# 页签
PAGE_START = 1
PAGE_END = 100

# 收藏值
COLLECTION_THRESHOLD = 500
R18_COLLECTION_THRESHOLD = 500
# endregion

# region 搜索地址
URL_SEARCH = 'https://www.pixiv.net/ajax/search/artworks/{0}?word={1}&p={2}'
URL_DETAIL = 'https://www.pixiv.net/bookmark_detail.php?illust_id={0}'
# endregion


class CPixivSearch(reptile_base.CReptileBase):


