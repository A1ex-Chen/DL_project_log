def ask(self, text, conv):
    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0
        ] and conv.messages[-1][1][-6:] == '</Img>':
        conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
    else:
        conv.append_message(conv.roles[0], text)
