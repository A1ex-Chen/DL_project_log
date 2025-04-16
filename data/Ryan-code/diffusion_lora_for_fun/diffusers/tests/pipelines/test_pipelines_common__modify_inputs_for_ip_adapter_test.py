def _modify_inputs_for_ip_adapter_test(self, inputs: Dict[str, Any]):
    parameters = inspect.signature(self.pipeline_class.__call__).parameters
    if 'image' in parameters.keys() and 'strength' in parameters.keys():
        inputs['num_inference_steps'] = 4
    inputs['output_type'] = 'np'
    inputs['return_dict'] = False
    return inputs
