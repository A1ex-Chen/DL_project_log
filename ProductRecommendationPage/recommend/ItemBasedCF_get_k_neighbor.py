def get_k_neighbor(self, k, vector1):
    nearest_books = dict()
    try:
        conn = MySQLdb.connect(host='localhost', user='root', passwd=
            '123456', db='blog', port=3306)
        cur = conn.cursor()
        cur.execute('select * from book_tag')
        results = cur.fetchall()
        all_booktag = dict()
        for row in results:
            book_id = row[1]
            book_tag = row[2:7]
            bt = [(0) for i in range(0, 15)]
            for i in range(0, 5):
                bt[book_tag[i] - 1] = 1
            cur_dis = cos_distance(vector1, bt)
            nearest_books[book_id] = cur_dis
        cur.close()
        conn.close()
    except (MySQLdb.Error, e):
        print('Mysql Error %d: %s' % (e.args[0], e.args[1]))
    return sorted(nearest_books.items(), lambda x, y: cmp(x[1], y[1]),
        reverse=True)[1:k + 1]
