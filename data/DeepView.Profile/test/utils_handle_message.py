def handle_message(self, message):
    from_server = innpv_pb2.FromServer()
    from_server.ParseFromString(message)
    print('From Server:')
    print(from_server)
    self.received_messages.append(from_server)
    print(f'new message. total: {len(self.received_messages)}')
