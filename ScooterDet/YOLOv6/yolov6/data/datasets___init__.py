def __init__(self, path, webcam, webcam_addr):
    self.webcam = webcam
    self.webcam_addr = webcam_addr
    if webcam:
        imgp = []
        vidp = [int(webcam_addr) if webcam_addr.isdigit() else webcam_addr]
    else:
        p = str(Path(path).resolve())
        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '**/*.*'), recursive=True)
                )
        elif os.path.isfile(p):
            files = [p]
        else:
            raise FileNotFoundError(f'Invalid path {p}')
        imgp = [i for i in files if i.split('.')[-1] in IMG_FORMATS]
        vidp = [v for v in files if v.split('.')[-1] in VID_FORMATS]
    self.files = imgp + vidp
    self.nf = len(self.files)
    self.type = 'image'
    if len(vidp) > 0:
        self.add_video(vidp[0])
    else:
        self.cap = None
