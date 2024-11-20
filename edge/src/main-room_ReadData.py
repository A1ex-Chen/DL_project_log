def ReadData(nameThread):
    global humidity
    global temperature
    global ppm
    print('Create thread read data')
    humidity = 0.0
    temperature = 0.0
    ppm = 0.0
    ser = serial.Serial(port='/dev/ttyACM0', baudrate=115200)
    time.sleep(8)
    while True:
        try:
            time.sleep(2)
            s = ser.readline()
            data = s.decode('utf-8')
            j = json.loads(data)
            humidity = j['humidity']
            temperature = j['temperature']
            ppm = j['ppm']
        except:
            print('error')
