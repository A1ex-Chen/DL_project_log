def get_comment(self, params):
    """
        获取对应商品id的评论
        :param params: 字典形式, 其中item_id为商品id, page为评论页码
        :return:
        """
    url = (
        'https://sclub.jd.com/comment/productPageComments.action?productId=%s&score=0&sortType=5&page=%d&pageSize=10'
         % (params['item_id'], params['page']))
    comment_headers = {'Referer': 'https://item.jd.com/%s.html' % params[
        'item_id'], 'User-Agent':
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
        }
    rsp = requests.get(url=url, headers=comment_headers).json()
    comments_count = rsp.get('productCommentSummary').get('commentCountStr')
    comments = rsp.get('comments')
    comments = [comment.get('content') for comment in comments]
