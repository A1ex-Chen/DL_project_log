@property
def modality_lengths(self):
    length_list = []
    for sample in self.list_data_dict:
        cur_len = sum(len(conv['value'].split()) for conv in sample[
            'conversations'])
        cur_len = cur_len if 'images' in sample else -cur_len
        length_list.append(cur_len)
    return length_list
