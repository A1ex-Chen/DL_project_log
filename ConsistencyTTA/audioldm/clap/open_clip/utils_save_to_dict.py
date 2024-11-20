def save_to_dict(s, o_={}):
    sp = s.split(': ')
    o_.update({sp[0]: float(sp[1])})
    return o_
