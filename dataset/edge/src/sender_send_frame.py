def send_frame(frame, humidity, temperature, ppm, count, channel=channel):
    start_time = time.time()
    send_frame = cv2.resize(frame, config.send_frame_reso)
    _, send_frame = cv2.imencode('.jpeg', send_frame)
    send_frame = send_frame.tobytes()
    image_byte = base64.b64encode(send_frame).decode('utf-8')
    data = {'ip': config.jetson_ip, 'image': str(image_byte), 'humidity':
        humidity, 'temperature': temperature, 'ppm': ppm, 'count': count,
        'type': 1}
    message = json.dumps(data)
    channel.basic_publish(exchange='', routing_key='hello', body=message)
    print('send frame time', time.time() - start_time, 's')
