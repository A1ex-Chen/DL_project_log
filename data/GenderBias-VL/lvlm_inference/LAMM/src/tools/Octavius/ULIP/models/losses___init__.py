def __init__(self, args):
    super().__init__()
    self.labels = None
    self.last_local_batch_size = None
    self.args = args
    if args.use_memory_bank:
        self.catfile = os.path.join('data/scanrefer', 'doc',
            'scanrefer_261_sorted.txt')
        self.obj_classes = [line.rstrip() for line in open(self.catfile)]
        bank_size = ['small', 'middle', 'big'][1]
        print(f'=> load image memory bank')
        with open(os.path.join('data/scanrefer/image_memory_bank',
            f'img_class_memory_bank_{bank_size}.pkl'), 'rb') as f:
            self.obj_class_memory_bank = pickle.load(f)
        print(f'=> load text memory bank')
        with open(os.path.join('data/scanrefer/text_memory_bank',
            f'text_memory_bank_{bank_size}.pkl'), 'rb') as f:
            self.text_memory_bank = pickle.load(f)
            self.text_memory_bank = {key.replace(' ', '_'): value for key,
                value in self.text_memory_bank.items()}
