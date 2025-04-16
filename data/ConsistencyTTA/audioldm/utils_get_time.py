def get_time():
    t = time.localtime()
    return time.strftime('%d_%m_%Y_%H_%M_%S', t)
