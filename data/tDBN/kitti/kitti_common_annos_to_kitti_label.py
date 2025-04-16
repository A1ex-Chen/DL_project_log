def annos_to_kitti_label(annos):
    num_instance = len(annos['name'])
    result_lines = []
    for i in range(num_instance):
        result_dict = {'name': annos['name'][i], 'truncated': annos[
            'truncated'][i], 'occluded': annos['occluded'][i], 'alpha':
            annos['alpha'][i], 'bbox': annos['bbox'][i], 'dimensions':
            annos['dimensions'][i], 'location': annos['location'][i],
            'rotation_y': annos['rotation_y'][i]}
        line = kitti_result_line(result_dict)
        result_lines.append(line)
    return result_lines
