def next_message_to_dict(object):
    message = next(object)
    return MessageToDict(message)
