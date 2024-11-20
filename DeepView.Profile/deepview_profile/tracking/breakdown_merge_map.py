@staticmethod
def merge_map(destination, to_merge):
    for key, value in to_merge.items():
        if key in destination:
            destination[key] += value
        else:
            destination[key] = value
