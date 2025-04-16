def from_list_format(self, list_format: List[Dict]):
    text = ''
    num_images = 0
    for ele in list_format:
        if 'image' in ele:
            num_images += 1
            text += f'Picture {num_images}: '
            text += self.image_start_tag + ele['image'] + self.image_end_tag
            text += '\n'
        elif 'text' in ele:
            text += ele['text']
        elif 'box' in ele:
            if 'ref' in ele:
                text += self.ref_start_tag + ele['ref'] + self.ref_end_tag
            for box in ele['box']:
                text += self.box_start_tag + '(%d,%d),(%d,%d)' % (box[0],
                    box[1], box[2], box[3]) + self.box_end_tag
        else:
            raise ValueError('Unsupport element: ' + str(ele))
    return text
