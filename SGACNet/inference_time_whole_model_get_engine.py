def get_engine(onnx_filepath, engine_filepath, trt_floatx=16, trt_batchsize
    =1, trt_workspace=2 << 30, force_rebuild=True):
    if not os.path.exists(engine_filepath) or force_rebuild:
        print('Building engine using onnx2trt')
        if trt_floatx == 32:
            print('... this may take a while')
        else:
            print('... this may take -> AGES <-')
        cmd = f'onnx2trt {onnx_filepath}'
        cmd += f' -d {trt_floatx}'
        cmd += f' -b {trt_batchsize}'
        cmd += f' -w {trt_workspace}'
        cmd += f' -o {engine_filepath}'
        try:
            print(cmd)
            out = subprocess.check_output(cmd, shell=True, stderr=
                subprocess.STDOUT, universal_newlines=True)
        except subprocess.CalledProcessError as e:
            print('onnx2trt failed:', e.returncode, e.output)
            raise
        print(out)
    print(f'Loading engine: {engine_filepath}')
    with open(engine_filepath, 'rb') as f, trt.Runtime(trt.Logger(trt.
        Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())
