def serialize_to_protobuf(self, entry):
    entry.run_time_ms = self.run_time_ms
    entry.size_bytes = self.size_bytes
    entry.invocations = self.invocations
