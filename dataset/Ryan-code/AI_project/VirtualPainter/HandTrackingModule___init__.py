def __init__(self, mode=False, maxHand=2, complexity=1, detectionCon=0.5,
    trackCon=0.5):
    self.mode = mode
    self.maxHands = maxHand
    self.complexity = complexity
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.
        complexity, self.detectionCon, self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils
    self.tipIds = [4, 8, 12, 16, 20]
