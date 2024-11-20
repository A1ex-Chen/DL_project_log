def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info(f'initializing ceph client ...')
        st = time.time()
        from petrel_client.client import Client
        client = Client(enable_mc=True)
        ed = time.time()
        logger.info(f'initialize client cost {ed - st:.2f} s')
