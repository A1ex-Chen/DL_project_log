@staticmethod
def hasNVTX(marker):
    if type(marker) is str:
        try:
            marker = eval(marker)
        except:
            return False
    if type(marker) is dict:
        keys = marker.keys()
        return 'mod' in keys and 'op' in keys and 'args' in keys
    else:
        return False
