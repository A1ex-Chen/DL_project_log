def ReadData(nameThread):
    global humidity
    global temperature
    global ppm
    print('Create thread read data')
    time.sleep(8)
    while True:
        try:
            time.sleep(2)
            humidity = random.random() * 100
            temperature = random.random() * 50
            ppm = random.random() * 400
        except KeyboardInterrupt:
            print('error')
