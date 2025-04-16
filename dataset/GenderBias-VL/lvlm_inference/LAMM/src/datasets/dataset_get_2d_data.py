def get_2d_data(self, index):

    def _get_elem_from_tree(tree, tag):
        return tree.getElementsByTagName(tag)[0].firstChild.data
    vision_paths = self.data_2d['vision_path_list'][index]
    output_texts = self.data_2d['caption_list'][index]
    task_type = self.data_2d['task_type_list'][index]
    if task_type == 'detection':
        for i in range(len(output_texts)):
            if output_texts[i]['from'] == 'gpt':
                xml_text = output_texts[i]['value']
                dom_tree = xml.dom.minidom.parse(io.StringIO(xml_text)
                    ).documentElement
                items = dom_tree.getElementsByTagName('Item')
                bboxes, clses = [], []
                for item in items:
                    x1 = clip(float(_get_elem_from_tree(item, 'x1')))
                    y1 = clip(float(_get_elem_from_tree(item, 'y1')))
                    x2 = clip(float(_get_elem_from_tree(item, 'x2')))
                    y2 = clip(float(_get_elem_from_tree(item, 'y2')))
                    cls = str(_get_elem_from_tree(item, 'cls'))
                    bboxes.append([x1, y1, x2, y2])
                    clses.append(cls)
                output_texts[i]['value'] = dict(bboxes=torch.tensor(bboxes),
                    clses=clses)
    return dict(vision_paths=vision_paths, output_texts=output_texts,
        task_type=task_type, vision_type='image')
