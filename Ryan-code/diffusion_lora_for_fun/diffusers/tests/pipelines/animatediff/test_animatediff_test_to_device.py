@unittest.skipIf(torch_device != 'cuda', reason=
    'CUDA and CPU are required to switch devices')
def test_to_device(self):
    components = self.get_dummy_components()
    pipe = self.pipeline_class(**components)
    pipe.set_progress_bar_config(disable=None)
    pipe.to('cpu')
    model_devices = [component.device.type for component in pipe.components
        .values() if hasattr(component, 'device')]
    self.assertTrue(all(device == 'cpu' for device in model_devices))
    output_cpu = pipe(**self.get_dummy_inputs('cpu'))[0]
    self.assertTrue(np.isnan(output_cpu).sum() == 0)
    pipe.to('cuda')
    model_devices = [component.device.type for component in pipe.components
        .values() if hasattr(component, 'device')]
    self.assertTrue(all(device == 'cuda' for device in model_devices))
    output_cuda = pipe(**self.get_dummy_inputs('cuda'))[0]
    self.assertTrue(np.isnan(to_np(output_cuda)).sum() == 0)
