def check_yolov5u_filename(file: str, verbose: bool=True):
    """Replace legacy YOLOv5 filenames with updated YOLOv5u filenames."""
    if 'yolov3' in file or 'yolov5' in file:
        if 'u.yaml' in file:
            file = file.replace('u.yaml', '.yaml')
        elif '.pt' in file and 'u' not in file:
            original_file = file
            file = re.sub('(.*yolov5([nsmlx]))\\.pt', '\\1u.pt', file)
            file = re.sub('(.*yolov5([nsmlx])6)\\.pt', '\\1u.pt', file)
            file = re.sub('(.*yolov3(|-tiny|-spp))\\.pt', '\\1u.pt', file)
            if file != original_file and verbose:
                LOGGER.info(
                    f"""PRO TIP 💡 Replace 'model={original_file}' with new 'model={file}'.
YOLOv5 'u' models are trained with https://github.com/ultralytics/ultralytics and feature improved performance vs standard YOLOv5 models trained with https://github.com/ultralytics/yolov5.
"""
                    )
    return file
