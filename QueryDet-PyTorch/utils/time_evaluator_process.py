def process(self, inputs, outputs):
    for output in outputs:
        if 'time' in output.keys():
            self.all_time.append(output['time'])
    return
