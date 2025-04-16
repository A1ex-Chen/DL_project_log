def _sent_lowercase(s):
    try:
        return s[0].lower() + s[1:]
    except:
        return s
