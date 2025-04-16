def benchmark(model=WEIGHTS_DIR / 'yolov8n.pt', data=None, imgsz=160, half=
    False, int8=False, device='cpu', verbose=False):
    """
    Benchmark a YOLO model across different formats for speed and accuracy.

    Args:
        model (str | Path | optional): Path to the model file or directory. Default is
            Path(SETTINGS['weights_dir']) / 'yolov8n.pt'.
        data (str, optional): Dataset to evaluate on, inherited from TASK2DATA if not passed. Default is None.
        imgsz (int, optional): Image size for the benchmark. Default is 160.
        half (bool, optional): Use half-precision for the model if True. Default is False.
        int8 (bool, optional): Use int8-precision for the model if True. Default is False.
        device (str, optional): Device to run the benchmark on, either 'cpu' or 'cuda'. Default is 'cpu'.
        verbose (bool | float | optional): If True or a float, assert benchmarks pass with given metric.
            Default is False.

    Returns:
        df (pandas.DataFrame): A pandas DataFrame with benchmark results for each format, including file size,
            metric, and inference time.

    Example:
        ```python
        from ultralytics.utils.benchmarks import benchmark

        benchmark(model='yolov8n.pt', imgsz=640)
        ```
    """
    import pandas as pd
    pd.options.display.max_columns = 10
    pd.options.display.width = 120
    device = select_device(device, verbose=False)
    if isinstance(model, (str, Path)):
        model = YOLO(model)
    is_end2end = getattr(model.model.model[-1], 'end2end', False)
    y = []
    t0 = time.time()
    for i, (name, format, suffix, cpu, gpu) in export_formats().iterrows():
        emoji, filename = '❌', None
        try:
            if i == 7:
                assert model.task != 'obb', 'TensorFlow GraphDef not supported for OBB task'
            elif i == 9:
                assert LINUX and not ARM64, 'Edge TPU export only supported on non-aarch64 Linux'
            elif i in {5, 10}:
                assert MACOS or LINUX, 'CoreML and TF.js export only supported on macOS and Linux'
                assert not IS_RASPBERRYPI, 'CoreML and TF.js export not supported on Raspberry Pi'
                assert not IS_JETSON, 'CoreML and TF.js export not supported on NVIDIA Jetson'
                assert not is_end2end, 'End-to-end models not supported by CoreML and TF.js yet'
            if i in {3, 5}:
                assert not IS_PYTHON_3_12, 'CoreML and OpenVINO not supported on Python 3.12'
            if i in {6, 7, 8, 9, 10}:
                assert not isinstance(model, YOLOWorld
                    ), 'YOLOWorldv2 TensorFlow exports not supported by onnx2tf yet'
                assert not is_end2end, 'End-to-end models not supported by onnx2tf yet'
            if i in {11}:
                assert not isinstance(model, YOLOWorld
                    ), 'YOLOWorldv2 Paddle exports not supported yet'
                assert not is_end2end, 'End-to-end models not supported by PaddlePaddle yet'
            if i in {12}:
                assert not isinstance(model, YOLOWorld
                    ), 'YOLOWorldv2 NCNN exports not supported yet'
                assert not is_end2end, 'End-to-end models not supported by NCNN yet'
            if 'cpu' in device.type:
                assert cpu, 'inference not supported on CPU'
            if 'cuda' in device.type:
                assert gpu, 'inference not supported on GPU'
            if format == '-':
                filename = model.ckpt_path or model.cfg
                exported_model = model
            else:
                filename = model.export(imgsz=imgsz, format=format, half=
                    half, int8=int8, device=device, verbose=False)
                exported_model = YOLO(filename, task=model.task)
                assert suffix in str(filename), 'export failed'
            emoji = '❎'
            assert model.task != 'pose' or i != 7, 'GraphDef Pose inference is not supported'
            assert i not in {9, 10}, 'inference not supported'
            assert i != 5 or platform.system(
                ) == 'Darwin', 'inference only supported on macOS>=10.13'
            exported_model.predict(ASSETS / 'bus.jpg', imgsz=imgsz, device=
                device, half=half)
            data = data or TASK2DATA[model.task]
            key = TASK2METRIC[model.task]
            results = exported_model.val(data=data, batch=1, imgsz=imgsz,
                plots=False, device=device, half=half, int8=int8, verbose=False
                )
            metric, speed = results.results_dict[key], results.speed[
                'inference']
            fps = round(1000 / speed, 2)
            y.append([name, '✅', round(file_size(filename), 1), round(
                metric, 4), round(speed, 2), fps])
        except Exception as e:
            if verbose:
                assert type(e
                    ) is AssertionError, f'Benchmark failure for {name}: {e}'
            LOGGER.warning(f'ERROR ❌️ Benchmark failure for {name}: {e}')
            y.append([name, emoji, round(file_size(filename), 1), None,
                None, None])
    check_yolo(device=device)
    df = pd.DataFrame(y, columns=['Format', 'Status❔', 'Size (MB)', key,
        'Inference time (ms/im)', 'FPS'])
    name = Path(model.ckpt_path).name
    s = f"""
Benchmarks complete for {name} on {data} at imgsz={imgsz} ({time.time() - t0:.2f}s)
{df}
"""
    LOGGER.info(s)
    with open('benchmarks.log', 'a', errors='ignore', encoding='utf-8') as f:
        f.write(s)
    if verbose and isinstance(verbose, float):
        metrics = df[key].array
        floor = verbose
        assert all(x > floor for x in metrics if pd.notna(x)
            ), f'Benchmark failure: metric(s) < floor {floor}'
    return df
