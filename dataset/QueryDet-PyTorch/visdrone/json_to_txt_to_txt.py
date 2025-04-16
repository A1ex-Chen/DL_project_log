def to_txt(self):
    for img_id in tqdm.tqdm(self.images.keys()):
        file_name = self.images[img_id]['file'].replace('jpg', 'txt')
        with open(os.path.join(self.out_dir, file_name), 'w') as fw:
            for pred in self.results[img_id]:
                row = '%.2f,%.2f,%.2f,%.2f,%.8f,%d,-1,-1' % (pred['box'][0],
                    pred['box'][1], pred['box'][2], pred['box'][3], pred[
                    'score'], pred['category'])
                fw.write(row + '\n')
