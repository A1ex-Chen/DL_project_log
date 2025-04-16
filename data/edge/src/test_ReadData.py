def ReadData(nameThread):
    global humidity
    global temperature
    global gas
    ser = serial.Serial(port='/dev/ttyACM0', baurate=9600)
    try:
        while True:
            s = ser.readline()
            data = s.decode('utf-8')
            j = json.loads(data)
            humidity = j['humidity']
            temperature = j['temperature']
            gas = j['gas']
            number = j['code']
            user = requests.get('http://{0}:8800/accounts/bycode/{1}'.
                format(config.server_ip, number))
            print(user.json())
            ser.write('{0}'.format(user.name))
    except KeyboardInterrupt:
        print('error')
