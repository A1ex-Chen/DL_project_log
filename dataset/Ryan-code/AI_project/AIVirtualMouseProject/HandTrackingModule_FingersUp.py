def FingersUp(self):
    fingers = []
    if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    for id in range(1, 5):
        if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2
            ]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers
