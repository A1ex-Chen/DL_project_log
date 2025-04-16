def __init__(self, input_path, save_path='.', information_file=None,
    make_video=True):
    if information_file is None:
        information_file = metrics.InformationFile(file_path=os.path.join(
            input_path, 'seqinfo.ini'))
    if make_video:
        file_name = os.path.split(input_path)[1]
        fps = information_file.search(variable_name='frameRate')
        horizontal_resolution = information_file.search(variable_name='imWidth'
            )
        vertical_resolution = information_file.search(variable_name='imHeight')
        image_size = horizontal_resolution, vertical_resolution
        videos_folder = os.path.join(save_path, 'videos')
        if not os.path.exists(videos_folder):
            os.makedirs(videos_folder)
        video_path = os.path.join(videos_folder, file_name + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.file_name = file_name
        self.video = cv2.VideoWriter(video_path, fourcc, fps, image_size)
    self.length = information_file.search(variable_name='seqLength')
    self.input_path = input_path
    self.frame_number = 1
    self.image_extension = information_file.search('imExt')
    self.image_directory = information_file.search('imDir')
