def gen_dataloader(yaml_path, task='train', imgsz=640, workers=4):
    data_yaml = check_yaml(yaml_path)
    data = check_dataset(data_yaml)
    dataloader = create_dataloader(data[task], imgsz=imgsz, batch_size=1,
        stride=32, pad=0.5, single_cls=False, rect=False, workers=workers)[0]
    return dataloader
