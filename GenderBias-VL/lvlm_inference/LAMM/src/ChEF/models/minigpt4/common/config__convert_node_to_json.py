def _convert_node_to_json(self, node):
    container = OmegaConf.to_container(node, resolve=True)
    return json.dumps(container, indent=4, sort_keys=True)
