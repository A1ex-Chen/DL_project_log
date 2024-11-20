def getProxy(proxy_url):
    response = requests.get(proxy_url)
    proxies_list = response.text.split('\n')
    for proxy_str in proxies_list:
        try:
            proxy_json = json.loads(proxy_str)
            host = proxy_json['host']
            port = proxy_json['port']
            type = proxy_json['type']
            verify(host, port, type)
        except:
            pass
