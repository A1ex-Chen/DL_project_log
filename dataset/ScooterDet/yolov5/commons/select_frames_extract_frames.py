def extract_frames(input_video_path, output_folder, frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f'frame_{frame_count}.jpg'
                )
            cv2.imwrite(frame_path, frame)
        frame_count += 1
    cap.release()
