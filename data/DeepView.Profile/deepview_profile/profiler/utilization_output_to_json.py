def output_to_json(self, path_save_file):
    node_json = self._convert_node_to_dict(self._root_node)
    output = {'node': node_json, 'tensor_core': self._tensor_core_perc}
    with open(os.path.join(path_save_file, 'profiling_results.json'), 'wb'
        ) as f:
        f.write(orjson.dumps(output))
