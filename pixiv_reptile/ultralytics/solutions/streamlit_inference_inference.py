def inference():
    """Runs real-time object detection on video input using Ultralytics YOLOv8 in a Streamlit application."""
    import streamlit as st
    from ultralytics import YOLO
    menu_style_cfg = '<style>MainMenu {visibility: hidden;}</style>'
    main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Ultralytics YOLOv8 Streamlit Application
                    </h1></div>"""
    sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Experience real-time object detection on your webcam with the power of Ultralytics YOLOv8! 🚀</h4>
                    </div>"""
    st.set_page_config(page_title='Ultralytics Streamlit App', layout=
        'wide', initial_sidebar_state='auto')
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)
    with st.sidebar:
        logo = (
            'https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg'
            )
        st.image(logo, width=250)
    st.sidebar.title('User Configuration')
    source = st.sidebar.selectbox('Video', ('webcam', 'video'))
    vid_file_name = ''
    if source == 'video':
        vid_file = st.sidebar.file_uploader('Upload Video File', type=[
            'mp4', 'mov', 'avi', 'mkv'])
        if vid_file is not None:
            g = io.BytesIO(vid_file.read())
            vid_location = 'ultralytics.mp4'
            with open(vid_location, 'wb') as out:
                out.write(g.read())
            vid_file_name = 'ultralytics.mp4'
    elif source == 'webcam':
        vid_file_name = 0
    yolov8_model = st.sidebar.selectbox('Model', ('YOLOv8n', 'YOLOv8s',
        'YOLOv8m', 'YOLOv8l', 'YOLOv8x', 'YOLOv8n-Seg', 'YOLOv8s-Seg',
        'YOLOv8m-Seg', 'YOLOv8l-Seg', 'YOLOv8x-Seg', 'YOLOv8n-Pose',
        'YOLOv8s-Pose', 'YOLOv8m-Pose', 'YOLOv8l-Pose', 'YOLOv8x-Pose'))
    model = YOLO(f'{yolov8_model.lower()}.pt')
    class_names = list(model.names.values())
    selected_classes = st.sidebar.multiselect('Classes', class_names,
        default=class_names[:3])
    selected_ind = [class_names.index(option) for option in selected_classes]
    if not isinstance(selected_ind, list):
        selected_ind = list(selected_ind)
    conf_thres = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.25, 0.01
        )
    nms_thres = st.sidebar.slider('NMS Threshold', 0.0, 1.0, 0.45, 0.01)
    col1, col2 = st.columns(2)
    org_frame = col1.empty()
    ann_frame = col2.empty()
    fps_display = st.sidebar.empty()
    if st.sidebar.button('Start'):
        videocapture = cv2.VideoCapture(vid_file_name)
        if not videocapture.isOpened():
            st.error('Could not open webcam.')
        stop_button = st.button('Stop')
        prev_time = 0
        while videocapture.isOpened():
            success, frame = videocapture.read()
            if not success:
                st.warning(
                    'Failed to read frame from webcam. Please make sure the webcam is connected properly.'
                    )
                break
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            results = model(frame, conf=float(conf_thres), iou=float(
                nms_thres), classes=selected_ind)
            annotated_frame = results[0].plot()
            org_frame.image(frame, channels='BGR')
            ann_frame.image(annotated_frame, channels='BGR')
            if stop_button:
                videocapture.release()
                torch.cuda.empty_cache()
                st.stop()
            fps_display.metric('FPS', f'{fps:.2f}')
        videocapture.release()
    torch.cuda.empty_cache()
    cv2.destroyAllWindows()
