def __init__(self, args):
    self.args = args
    os.makedirs(os.path.join(exp_dir, self.args.sub_exp), exist_ok=True)
    logdir = os.path.join(exp_dir, self.args.sub_exp, 'logs')
    os.makedirs(logdir, exist_ok=True)
    logfile = args.log_file
    self.log = open(os.path.join(logdir, logfile), 'w')
    test_case_path = 'test_case.txt'
    self.test_case = open(os.path.join(logdir, test_case_path), 'w')
    multi_face_path = 'multi_face.txt'
    self.multi_face = open(os.path.join(logdir, multi_face_path), 'w')
    no_face_path = 'no_face.txt'
    self.no_face = open(os.path.join(logdir, no_face_path), 'w')
    nsfw_image_path = 'nsfw_image.txt'
    self.nsfw_image = open(os.path.join(logdir, nsfw_image_path), 'w')
    diff_box_path = 'diff_box.txt'
    self.diff_box = open(os.path.join(logdir, diff_box_path), 'w')
    self.multi_face_dir = os.path.join(exp_dir, self.args.sub_exp,
        'multi_face_imgs')
    os.makedirs(self.multi_face_dir, exist_ok=True)
    self.no_face_dir = os.path.join(exp_dir, self.args.sub_exp, 'no_face_imgs')
    os.makedirs(self.no_face_dir, exist_ok=True)
    self.nsfw_image_dir = os.path.join(exp_dir, self.args.sub_exp, 'nsfw_imgs')
    os.makedirs(self.nsfw_image_dir, exist_ok=True)
    self.diff_box_dir = os.path.join(exp_dir, self.args.sub_exp,
        'diff_box_imgs')
    os.makedirs(self.diff_box_dir, exist_ok=True)
    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    self.setup_seed()
    self.build_dataset()
    self.build_model()
