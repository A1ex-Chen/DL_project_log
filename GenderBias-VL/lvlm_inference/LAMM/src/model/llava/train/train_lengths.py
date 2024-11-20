@property
def lengths(self):
    length_list = []
    for sample in self.list_data_dict:
        img_tokens = 128 if 'image' in sample else 0
        length_list.append(sum(len(conv['value'].split()) for conv in
            sample['conversations']) + img_tokens)
    return length_list
