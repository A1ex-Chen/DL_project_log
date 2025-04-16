def save_boxes(self, path: (str | Path)):
    with open(path, 'w') as f:
        for box in self.preds[:, :4]:
            f.write(','.join(map(str, box)) + '\n')
