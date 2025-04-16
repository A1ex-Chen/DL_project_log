def print(self):
    self.display(pprint=True)
    print(
        f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}'
         % self.t)
