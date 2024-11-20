def itemToMysql(value):
    db = pymysql.connect(host='localhost', user='root', password='root',
        port=3306, db='curdesign', charset='gbk')
    cursor = db.cursor()
    sql = (
        'INSERT INTO items_item(item_id, title, price, pic_file, pic_url, type, sales) VALUES (%s, %s, %s, %s, %s, %s, %s)'
        )
    try:
        cursor.execute(sql, (value[0], value[1], value[2], value[3], value[
            4], value[5], value[6]))
        db.commit()
    except:
        try:
            cursor.execute(sql, (value[0], value[1], value[2], value[3],
                value[4], value[5], value[6]))
            db.commit()
        except:
            db.rollback()
            print('保存失败')
    cursor.close()
    db.close()
