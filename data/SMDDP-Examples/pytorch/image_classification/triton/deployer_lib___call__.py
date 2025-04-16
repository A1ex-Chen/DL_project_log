def __call__(self, *inputs):
    inp = [(input_name, inputs[i]) for i, input_name in enumerate(self.
        input_names)]
    inp = {input_name: self.to_numpy(x) for input_name, x in inp}
    outputs = self.session.run(None, inp)
    outputs = [torch.from_numpy(output) for output in outputs]
    outputs = [output.to(device) for output in outputs]
    if len(outputs) == 1:
        outputs = outputs[0]
    return outputs
