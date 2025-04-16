def _configure_libraries():
    """
    Configurations for some libraries.
    """
    disable_cv2 = int(os.environ.get('DETECTRON2_DISABLE_CV2', False))
    if disable_cv2:
        sys.modules['cv2'] = None
    else:
        os.environ['OPENCV_OPENCL_RUNTIME'] = 'disabled'
        try:
            import cv2
            if int(cv2.__version__.split('.')[0]) >= 3:
                cv2.ocl.setUseOpenCL(False)
        except ModuleNotFoundError:
            pass

    def get_version(module, digit=2):
        return tuple(map(int, module.__version__.split('.')[:digit]))
    assert get_version(torch) >= (1, 4), 'Requires torch>=1.4'
    import fvcore
    assert get_version(fvcore, 3) >= (0, 1, 2), 'Requires fvcore>=0.1.2'
    import yaml
    assert get_version(yaml) >= (5, 1), 'Requires pyyaml>=5.1'
