def get_tag_vec(self, book_id):
    try:
        conn = MySQLdb.connect(host='localhost', user='root', passwd=
            '123456', db='blog', port=3306)
        cur = conn.cursor()
        sql = 'select * from book_tag where book_id=' + str(book_id)
        cur.execute(sql)
        cur_book_tag = cur.fetchone()[2:7]
        cur.close()
        conn.close()
        bt = [(0) for i in range(0, 15)]
        for i in range(0, 5):
            bt[cur_book_tag[i] - 1] = 1
        return bt
    except (MySQLdb.Error, e):
        print('Mysql Error %d: %s' % (e.args[0], e.args[1]))
