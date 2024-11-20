def main(self, index_pn, comment_pn):
    """
        实现爬取的函数
        :param index_pn: 爬取搜索页的页码总数
        :param comment_pn: 爬取评论页的页码总数
        :return:
        """
    il = [(i * 2 + 1) for i in range(index_pn)]
    with futures.ThreadPoolExecutor(15) as executor:
        res = executor.map(self.get_index, il)
    for item_ids in res:
        cl = [{'item_id': item_id, 'page': page} for item_id in item_ids for
            page in range(comment_pn)]
        with futures.ThreadPoolExecutor(15) as executor:
            executor.map(self.get_comment, cl)
