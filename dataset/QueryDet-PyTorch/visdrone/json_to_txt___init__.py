def __init__(self, gt_json, det_json, out_dir):
    gt_data = json.load(open(gt_json))
    self.images = {x['id']: {'file': x['file_name'], 'height': x['height'],
        'width': x['width']} for x in gt_data['images']}
    det_data = json.load(open(det_json))
    self.results = {}
    for result in det_data:
        if result['image_id'] not in self.results.keys():
            self.results[result['image_id']] = []
        self.results[result['image_id']].append({'box': result['bbox'],
            'category': result['category_id'], 'score': result['score']})
    self.out_dir = out_dir
