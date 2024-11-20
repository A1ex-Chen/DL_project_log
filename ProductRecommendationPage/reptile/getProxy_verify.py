def verify(ip, port, type):
    proxies = {}
    try:
        telnet = telnetlib.Telnet(ip, port=port, timeout=3)
    except:
        print('unconnected')
    else:
        proxies['type'] = type
        proxies['host'] = ip
        proxies['port'] = port
        proxiesJson = json.dumps(proxies)
        with open('./verified_proxies.json', 'a+') as f:
            f.write(proxiesJson + '\n')
        print('已写入：%s' % proxies)
