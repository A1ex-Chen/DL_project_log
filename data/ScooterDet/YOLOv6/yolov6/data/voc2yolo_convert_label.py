def convert_label(path, lb_path, year, image_id):

    def convert_box(size, box):
        dw, dh = 1.0 / size[0], 1.0 / size[1]
        x, y, w, h = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]
            ) / 2.0 - 1, box[1] - box[0], box[3] - box[2]
        return x * dw, y * dh, w * dw, h * dh
    in_file = open(os.path.join(path, f'VOC{year}/Annotations/{image_id}.xml'))
    out_file = open(lb_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls in VOC_NAMES and not int(obj.find('difficult').text) == 1:
            xmlbox = obj.find('bndbox')
            bb = convert_box((w, h), [float(xmlbox.find(x).text) for x in (
                'xmin', 'xmax', 'ymin', 'ymax')])
            cls_id = VOC_NAMES.index(cls)
            out_file.write(' '.join([str(a) for a in (cls_id, *bb)]) + '\n')
