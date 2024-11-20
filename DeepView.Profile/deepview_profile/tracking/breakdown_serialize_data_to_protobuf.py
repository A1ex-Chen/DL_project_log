def serialize_data_to_protobuf(self, entry):
    entry.weight.size_bytes = self.size_bytes
    entry.weight.grad_size_bytes = self.grad_size_bytes
