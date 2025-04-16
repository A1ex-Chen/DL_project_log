def __init__(self, keyword):
    """
        初始化
        :param keyword: 搜索的关键词
        """
    self.keyword = keyword
    self.mongo_client = pymongo.MongoClient(host='localhost')
    self.mongo_collection = self.mongo_client['spiders']['jd']
    self.mongo_collection.create_index([('item_id', pymongo.ASCENDING)])
