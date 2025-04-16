def to_dict(self, sample):
    return {'image': sample[0], 'answer': self.text_processor(sample[1][
        'caption'])}
