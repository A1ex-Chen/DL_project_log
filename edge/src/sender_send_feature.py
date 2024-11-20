def send_feature(tracked_objects, channel=channel):
    start_time = time.time()
    for o in tracked_objects:
        data = {'ip': config.jetson_ip, 'userId': o.id, 'position': base64.
            binascii.b2a_base64(np.float64(np.array(o.last_detection.data))
            ).decode('ascii'), 'type': 2}
        if o.last_detection.embedding is not None:
            data['vector'] = base64.binascii.b2a_base64(np.float64(o.
                last_detection.embedding)).decode('ascii')
        message = json.dumps(data)
        channel.basic_publish(exchange='', routing_key='q-2', body=message)
    print('send features time', time.time() - start_time, 's')
