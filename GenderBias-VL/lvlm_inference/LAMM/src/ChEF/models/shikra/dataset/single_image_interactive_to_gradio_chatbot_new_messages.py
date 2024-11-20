def to_gradio_chatbot_new_messages(self):
    conv = self.__getitem__(0, return_conv=True)
    new_messages = conv.messages[-2:]
    ret_messages = []
    for r, m in new_messages:
        nm = m.replace('<im_patch>', '').replace('<im_end>', '').replace(
            '<im_start>', '<image>')
        ret_messages.append((r, nm))
    return ret_messages
