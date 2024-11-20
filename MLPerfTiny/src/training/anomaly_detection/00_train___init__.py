def __init__(self):
    import matplotlib.pyplot as plt
    self.plt = plt
    self.fig = self.plt.figure(figsize=(30, 10))
    self.plt.subplots_adjust(wspace=0.3, hspace=0.3)
