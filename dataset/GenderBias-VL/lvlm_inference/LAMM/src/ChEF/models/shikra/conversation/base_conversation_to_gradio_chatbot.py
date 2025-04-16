def to_gradio_chatbot(self):
    """Convert the history to gradio chatbot format"""
    ret = []
    for i, (role, msg) in enumerate(self.messages[self.offset:]):
        if i % 2 == 0:
            ret.append([msg, None])
        else:
            ret[-1][-1] = msg
    return ret
