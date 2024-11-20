def load_all_allsides_triples(return_type='body'):
    with open('/home/nayeon/omission/data/crawled_article_lvl_3_ids.txt', 'r'
        ) as infile:
        allsides_news_ids = infile.read()
        allsides_news_ids = allsides_news_ids.split('\n')
    all_triples = []
    for n_id in allsides_news_ids:
        triple_obj = {'id': n_id}
        for leaning in ['left', 'right', 'center']:
            with open('/home/nayeon/omission/data/articles/{}_{}.json'.
                format(n_id, leaning), 'r') as in_json:
                json_obj = json.load(in_json)
                if return_type == 'body':
                    triple_obj[leaning] = ' '.join(json_obj['fullArticle'])
                elif return_type == 'title':
                    triple_obj[leaning] = json_obj['newsTitle']
        all_triples += triple_obj,
    return all_triples
