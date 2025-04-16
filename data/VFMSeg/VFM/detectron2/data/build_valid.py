def valid(anns):
    for ann in anns:
        if ann.get('iscrowd', 0) == 0:
            return True
    return False
