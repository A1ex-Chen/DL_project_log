from . import reptile_base
from . import ctrl_common
from tqdm import tqdm

# 爬取页数
PAGE_START = 1
PAGE_END = 2

# region 已关注用户的作品地址
DYNAMIC_URL = 'https://www.pixiv.net/ajax/follow_latest/illust?p={0}&mode=all&lang=zh'
# endregion

class CPixivDynamic(reptile_base.CReptileBase):
    # 【已关注用户的作品】

