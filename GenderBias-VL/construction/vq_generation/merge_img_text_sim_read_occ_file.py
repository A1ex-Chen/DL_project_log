def read_occ_file():
    path = '../../resources/occ_us.csv'
    data_class = {}
    with open(path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            job = row['occupation']
            first_class = row['First Class']
            second_class = row['Second Class']
            third_class = row['Third Class']
            label_class = second_class
            if (first_class !=
                'Management, professional, and related occupations'):
                label_class = first_class
            data_class[job
                ] = first_class, second_class, third_class, label_class
    return data_class
