def alive(self):
    return not self.received_messages or not self.received_messages[-1
        ].HasField('analysis_error')
