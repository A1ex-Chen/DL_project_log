def resitemToMysql(value):
    db = pymysql.connect(host='localhost', user='root', password='root',
        port=3306, db='curdesign', charset='gbk')
    cursor = db.cursor()
    sql_create = (
        'INSERT INTO analytics_rec_items(user_id, rec_item) VALUES (%s,%s)')
    sql_update = (
        'UPDATE analytics_rec_items SET rec_item = %s WHERE user_id = %s')
    try:
        cursor.execute(sql_create, (value[0], value[1]))
        db.commit()
    except:
        try:
            cursor.execute(sql_update, (value[1], value[0]))
            db.commit()
        except:
            db.rollback()
            print('保存失败')
    cursor.close()
    db.close()
