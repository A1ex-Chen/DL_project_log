def collect_env_info():
    has_gpu = torch.cuda.is_available()
    torch_version = torch.__version__
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME
    has_rocm = False
    if getattr(torch.version, 'hip', None
        ) is not None and ROCM_HOME is not None:
        has_rocm = True
    has_cuda = has_gpu and not has_rocm
    data = []
    data.append(('sys.platform', sys.platform))
    data.append(('Python', sys.version.replace('\n', '')))
    data.append(('numpy', np.__version__))
    try:
        import detectron2
        data.append(('detectron2', detectron2.__version__ + ' @' + os.path.
            dirname(detectron2.__file__)))
    except ImportError:
        data.append(('detectron2', 'failed to import'))
    except AttributeError:
        data.append(('detectron2', 'imported a wrong installation'))
    try:
        import detectron2._C as _C
    except ImportError as e:
        data.append(('detectron2._C', f'not built correctly: {e}'))
        if sys.platform != 'win32':
            try:
                cxx = os.environ.get('CXX', 'c++')
                cxx = subprocess.check_output("'{}' --version".format(cxx),
                    shell=True)
                cxx = cxx.decode('utf-8').strip().split('\n')[0]
            except subprocess.SubprocessError:
                cxx = 'Not found'
            data.append(('Compiler ($CXX)', cxx))
            if has_cuda and CUDA_HOME is not None:
                try:
                    nvcc = os.path.join(CUDA_HOME, 'bin', 'nvcc')
                    nvcc = subprocess.check_output("'{}' -V".format(nvcc),
                        shell=True)
                    nvcc = nvcc.decode('utf-8').strip().split('\n')[-1]
                except subprocess.SubprocessError:
                    nvcc = 'Not found'
                data.append(('CUDA compiler', nvcc))
        if has_cuda and sys.platform != 'win32':
            try:
                so_file = importlib.util.find_spec('detectron2._C').origin
            except (ImportError, AttributeError):
                pass
            else:
                data.append(('detectron2 arch flags',
                    detect_compute_compatibility(CUDA_HOME, so_file)))
    else:
        data.append(('Compiler', _C.get_compiler_version()))
        data.append(('CUDA compiler', _C.get_cuda_version()))
        if has_cuda and getattr(_C, 'has_cuda', lambda : True)():
            data.append(('detectron2 arch flags',
                detect_compute_compatibility(CUDA_HOME, _C.__file__)))
    data.append(get_env_module())
    data.append(('PyTorch', torch_version + ' @' + os.path.dirname(torch.
        __file__)))
    data.append(('PyTorch debug build', torch.version.debug))
    if not has_gpu:
        has_gpu_text = 'No: torch.cuda.is_available() == False'
    else:
        has_gpu_text = 'Yes'
    data.append(('GPU available', has_gpu_text))
    if has_gpu:
        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            cap = '.'.join(str(x) for x in torch.cuda.get_device_capability(k))
            name = torch.cuda.get_device_name(k) + f' (arch={cap})'
            devices[name].append(str(k))
        for name, devids in devices.items():
            data.append(('GPU ' + ','.join(devids), name))
        if has_rocm:
            msg = ' - invalid!' if not (ROCM_HOME and os.path.isdir(ROCM_HOME)
                ) else ''
            data.append(('ROCM_HOME', str(ROCM_HOME) + msg))
        else:
            try:
                from torch.utils.collect_env import get_nvidia_driver_version, run as _run
                data.append(('Driver version', get_nvidia_driver_version(_run))
                    )
            except Exception:
                pass
            msg = ' - invalid!' if not (CUDA_HOME and os.path.isdir(CUDA_HOME)
                ) else ''
            data.append(('CUDA_HOME', str(CUDA_HOME) + msg))
            cuda_arch_list = os.environ.get('TORCH_CUDA_ARCH_LIST', None)
            if cuda_arch_list:
                data.append(('TORCH_CUDA_ARCH_LIST', cuda_arch_list))
    data.append(('Pillow', PIL.__version__))
    try:
        data.append(('torchvision', str(torchvision.__version__) + ' @' +
            os.path.dirname(torchvision.__file__)))
        if has_cuda:
            try:
                torchvision_C = importlib.util.find_spec('torchvision._C'
                    ).origin
                msg = detect_compute_compatibility(CUDA_HOME, torchvision_C)
                data.append(('torchvision arch flags', msg))
            except (ImportError, AttributeError):
                data.append(('torchvision._C', 'Not found'))
    except AttributeError:
        data.append(('torchvision', 'unknown'))
    try:
        import fvcore
        data.append(('fvcore', fvcore.__version__))
    except (ImportError, AttributeError):
        pass
    try:
        import iopath
        data.append(('iopath', iopath.__version__))
    except (ImportError, AttributeError):
        pass
    try:
        import cv2
        data.append(('cv2', cv2.__version__))
    except (ImportError, AttributeError):
        data.append(('cv2', 'Not found'))
    env_str = tabulate(data) + '\n'
    env_str += collect_torch_env()
    return env_str
