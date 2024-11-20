def run(weights=ROOT / 'yolov5s.pt', imgsz=640, batch_size=1, data=ROOT /
    'data/coco128.yaml', device='', half=False, test=False, pt_only=False,
    hard_fail=False):
    y, t = [], time.time()
    device = select_device(device)
    for i, (name, f, suffix, cpu, gpu) in export.export_formats().iterrows():
        try:
            assert i not in (9, 10), 'inference not supported'
            assert i != 5 or platform.system(
                ) == 'Darwin', 'inference only supported on macOS>=10.13'
            if 'cpu' in device.type:
                assert cpu, 'inference not supported on CPU'
            if 'cuda' in device.type:
                assert gpu, 'inference not supported on GPU'
            if f == '-':
                w = weights
            else:
                w = export.run(weights=weights, imgsz=[imgsz], include=[f],
                    device=device, half=half)[-1]
            assert suffix in str(w), 'export failed'
            result = val.run(data, w, batch_size, imgsz, plots=False,
                device=device, task='benchmark', half=half)
            metrics = result[0]
            speeds = result[2]
            y.append([name, round(file_size(w), 1), round(metrics[3], 4),
                round(speeds[1], 2)])
        except Exception as e:
            if hard_fail:
                assert type(e
                    ) is AssertionError, f'Benchmark --hard-fail for {name}: {e}'
            LOGGER.warning(f'WARNING: Benchmark failure for {name}: {e}')
            y.append([name, None, None, None])
        if pt_only and i == 0:
            break
    LOGGER.info('\n')
    parse_opt()
    notebook_init()
    c = ['Format', 'Size (MB)', 'mAP@0.5:0.95', 'Inference time (ms)'
        ] if map else ['Format', 'Export', '', '']
    py = pd.DataFrame(y, columns=c)
    LOGGER.info(f'\nBenchmarks complete ({time.time() - t:.2f}s)')
    LOGGER.info(str(py if map else py.iloc[:, :2]))
    return py
