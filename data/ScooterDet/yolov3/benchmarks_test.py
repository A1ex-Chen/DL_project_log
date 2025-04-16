def test(weights=ROOT / 'yolov5s.pt', imgsz=640, batch_size=1, data=ROOT /
    'data/coco128.yaml', device='', half=False, test=False, pt_only=False,
    hard_fail=False):
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, gpu) in export.export_formats().iterrows():
        try:
            w = weights if f == '-' else export.run(weights=weights, imgsz=
                [imgsz], include=[f], device=device, half=half)[-1]
            assert suffix in str(w), 'export failed'
            y.append([name, True])
        except Exception:
            y.append([name, False])
    LOGGER.info('\n')
    parse_opt()
    notebook_init()
    py = pd.DataFrame(y, columns=['Format', 'Export'])
    LOGGER.info(f'\nExports complete ({time.time() - t:.2f}s)')
    LOGGER.info(str(py))
    return py
