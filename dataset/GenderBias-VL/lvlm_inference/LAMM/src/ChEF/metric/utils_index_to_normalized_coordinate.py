def index_to_normalized_coordinate(index):
    row = index // 32
    col = index % 32
    normalized_y = row / 32
    normalized_x = col / 32
    return normalized_x, normalized_y
