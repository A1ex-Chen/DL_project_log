def batch_from_throughput(self, throughput):
    throughput_ms = throughput / 1000
    return throughput_ms * self.runtime_model_ms.bias / (1 - throughput_ms *
        self.runtime_model_ms.coefficient)
