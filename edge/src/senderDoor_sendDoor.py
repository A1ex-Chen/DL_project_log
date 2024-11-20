def sendDoor(tracked_objects, number, ori_shape):
    haveEmbedding = False
    print('Creating connection...')
    url = os.environ.get('CLOUDAMQP_URL',
        f'amqp://admin:admin@{config.server_ip}:5672')
    params = pika.URLParameters(url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue='q-3')
    print('Connection established')
    LOGGER.info('Sender')
    start_time = time.time()
    for o in tracked_objects:
        if o.last_detection.embedding is not None:
            print(np.array(o.last_detection.data))
            if check_position(np.array(o.last_detection.data), ori_shape):
                data = {'ip': config.jetson_ip, 'userId': o.id, 'code':
                    number, 'position': base64.binascii.b2a_base64(np.
                    float64(np.array(o.last_detection.data))).decode(
                    'ascii'), 'vector': base64.binascii.b2a_base64(np.
                    float64(o.last_detection.embedding)).decode('ascii'),
                    'type': 3}
                message = json.dumps(data)
                channel.basic_publish(exchange='', routing_key='q-2', body=
                    message)
                haveEmbedding = True
                print('send features time', time.time() - start_time, 's')
                break
    connection.close()
    return haveEmbedding
