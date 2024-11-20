def require_onnxruntime(test_case):
    """
    Decorator marking a test that requires onnxruntime. These tests are skipped when onnxruntime isn't installed.
    """
    return unittest.skipUnless(is_onnx_available(), 'test requires onnxruntime'
        )(test_case)
