def forward(self, input1: Bunch):
    with lock:
        print('input1 =', input1)
    return input1
