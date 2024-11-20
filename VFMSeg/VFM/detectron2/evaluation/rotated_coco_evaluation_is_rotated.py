@staticmethod
def is_rotated(box_list):
    if type(box_list) == np.ndarray:
        return box_list.shape[1] == 5
    elif type(box_list) == list:
        if box_list == []:
            return False
        return np.all(np.array([(len(obj) == 5 and (type(obj) == list or 
            type(obj) == np.ndarray)) for obj in box_list]))
    return False
