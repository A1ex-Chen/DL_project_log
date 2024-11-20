def save_graph(self, output_file, inputs=None):
    """
        Save the graph as SVG format.

        Args:
            output_file (str): a SVG file
            inputs: optional inputs given to the model.
                If given, the inputs will be used to run the graph to record
                shape of every tensor. The shape information will be
                saved together with the graph.
        """
    from .caffe2_export import run_and_save_graph
    if inputs is None:
        save_graph(self._predict_net, output_file, op_only=False)
    else:
        size_divisibility = get_pb_arg_vali(self._predict_net,
            'size_divisibility', 0)
        device = get_pb_arg_vals(self._predict_net, 'device', b'cpu').decode(
            'ascii')
        inputs = convert_batched_inputs_to_c2_format(inputs,
            size_divisibility, device)
        inputs = [x.cpu().numpy() for x in inputs]
        run_and_save_graph(self._predict_net, self._init_net, inputs,
            output_file)
