import json
from tqdm import tqdm
from . import reptile_base
from . import ctrl_common


# region 画师信息
PAINTER_ID = 101490306
# endregion

# region 画师地址信息
URL_PAINTER_ILLUSTS_ID = 'https://www.pixiv.net/ajax/user/{0}/profile/all?lang=zh'
URL_PAINTER_ILLUSTS = 'https://www.pixiv.net/ajax/user/{0}/profile/illusts?{1}work_category=illustManga&is_first_page=1&lang=zh'
IDS = 'ids%5B%5D={0}&'
# endregion

class CPixivPainter(reptile_base.CReptileBase):
    # 【指定画师】

