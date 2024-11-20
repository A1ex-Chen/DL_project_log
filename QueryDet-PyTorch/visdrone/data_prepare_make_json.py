def make_json(images, annotations, new_label_json):
    ann_dict = {}
    ann_dict['categories'] = [{'supercategory': 'things', 'id': 1, 'name':
        'pedestrian'}, {'supercategory': 'things', 'id': 2, 'name':
        'people'}, {'supercategory': 'things', 'id': 3, 'name': 'bicycle'},
        {'supercategory': 'things', 'id': 4, 'name': 'car'}, {
        'supercategory': 'things', 'id': 5, 'name': 'van'}, {
        'supercategory': 'things', 'id': 6, 'name': 'truck'}, {
        'supercategory': 'things', 'id': 7, 'name': 'tricycle'}, {
        'supercategory': 'things', 'id': 8, 'name': 'awning-tricycle'}, {
        'supercategory': 'things', 'id': 9, 'name': 'bus'}, {
        'supercategory': 'things', 'id': 10, 'name': 'motor'}]
    ann_dict['images'] = images
    ann_dict['annotations'] = annotations
    with open(new_label_json, 'w') as outfile:
        json.dump(ann_dict, outfile)
