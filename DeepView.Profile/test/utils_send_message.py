def send_message(self, message):
    msg = innpv_pb2.FromClient()
    msg.sequence_number = self.seq_num
    self.seq_num += 1
    if isinstance(message, innpv_pb2.InitializeRequest):
        msg.initialize.CopyFrom(message)
    elif isinstance(message, innpv_pb2.AnalysisRequest):
        msg.analysis.CopyFrom(message)
    buf = msg.SerializeToString()
    length_buffer = struct.pack('>I', len(buf))
    self.socket.sendall(length_buffer)
    self.socket.sendall(buf)
