def uncapitalize(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ''
