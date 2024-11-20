
# region P战地址
PIXIV_URL = 'https://www.pixiv.net/'
# endregion

# region 爬虫代理头
REPTILE_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0'
# endregion

# region Cookies
PIXIV_COOKIES = '51556051_sZcQzuh69nSAjpr1ypFEgG3ggp0wbkt9'  # 请自行添加！！！！！！！！！
COOKIE_HEAD = 'PHPSESSID='
# endregion

# region 线程数量
NUM_THREAD = 10     # 下载线程数量
MANGE_NUM_THREAD = 5  # 多图下载线程数量
# endregion

# region 重试次数
RESTART_TIMES = 3
# endregion

# region Log上限数量
LOG_MAX_LENGTH = 50
# endregion

# region 下载图片地址
PICTURE_URL = 'https://www.pixiv.net/member_illust.php?mode=medium&illust_id={0}'
PICTURE_AJAX_URL = 'https://www.pixiv.net/touch/ajax/illust/details?illust_id={0}'
# endregion

# region 保存地址
SAVE_PATH = 'F:\\图集\\'
# endregion

# region 数据库信息
PICTURES_PATH = f'{SAVE_PATH}_PicturesDB/'
PICTURES_DB = '{0}.db'
# endregion