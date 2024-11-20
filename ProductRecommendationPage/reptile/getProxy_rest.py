import json
import telnetlib
import requests
import random

proxy_url = 'https://raw.githubusercontent.com/fate0/proxylist/master/proxy.list'
# proxyList = []




if __name__ == '__main__':
    getProxy(proxy_url)