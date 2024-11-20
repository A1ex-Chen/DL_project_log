def ReadData():
    ser = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
    try:
        while True:
            s = ser.readline()
            data = s.decode('utf-8')
            j = json.loads(data)
            number = j['code']
            user = requests.get('http://{0}:8800/accounts/bycode/{1}'.
                format(rabbitmq, number))
            print(user.json())
            name = user.json()['name']
            ser.write(name.encode())
            break
    except KeyboardInterrupt:
        print('error')
