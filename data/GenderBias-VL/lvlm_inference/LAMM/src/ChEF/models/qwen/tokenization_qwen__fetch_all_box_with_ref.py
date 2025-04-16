def _fetch_all_box_with_ref(self, text):
    list_format = self.to_list_format(text)
    output = []
    for i, ele in enumerate(list_format):
        if 'box' in ele:
            bbox = tuple(map(int, ele['box'].replace('(', '').replace(')',
                '').split(',')))
            assert len(bbox) == 4
            output.append({'box': bbox})
            if i > 0 and 'ref' in list_format[i - 1]:
                output[-1]['ref'] = list_format[i - 1]['ref'].strip()
    return output
