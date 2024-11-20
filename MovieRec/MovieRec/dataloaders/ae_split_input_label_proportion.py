def split_input_label_proportion(self, data, label_prop=0.2):
    input_list, label_list = [], []
    for items in data.values():
        items = np.array(items)
        if len(items) * label_prop >= 1:
            choose_as_label = np.zeros(len(items), dtype='bool')
            chosen_index = np.random.choice(len(items), size=int(label_prop *
                len(items)), replace=False).astype('int64')
            choose_as_label[chosen_index] = True
            input_list.append(items[np.logical_not(choose_as_label)])
            label_list.append(items[choose_as_label])
        else:
            input_list.append(items)
            label_list.append(np.array([]))
    return input_list, label_list
