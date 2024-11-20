def get_class_to_label_map():
    class_to_label = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Van': 3,
        'Person_sitting': 4, 'Truck': 5, 'Tram': 6, 'Misc': 7, 'DontCare': -1}
    return class_to_label
