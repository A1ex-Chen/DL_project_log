def run_models(self, models, inputs):
    """ run the models on inputs, return the outputs and execution times """
    ret = []
    for model in models:
        torch.cuda.synchronize()
        time_start = time.time()
        outputs = []
        for input in inputs:
            with torch.no_grad():
                output = model(*input)
            if type(output) is torch.Tensor:
                output = [output]
            outputs.append(output)
        torch.cuda.synchronize()
        time_end = time.time()
        t = time_end - time_start
        ret.append(outputs)
        ret.append(t)
    return ret
