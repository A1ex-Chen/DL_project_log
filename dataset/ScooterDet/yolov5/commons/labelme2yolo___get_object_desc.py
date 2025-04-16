def __get_object_desc(obj_port_list):
    __get_dist = lambda int_list: max(int_list) - min(int_list)
    x_lists = [port[0] for port in obj_port_list]
    y_lists = [port[1] for port in obj_port_list]
    return min(x_lists), __get_dist(x_lists), min(y_lists), __get_dist(y_lists)
