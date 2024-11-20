def now():
    from datetime import datetime
    return datetime.now().strftime('%Y%m%d%H%M')[:-1]
