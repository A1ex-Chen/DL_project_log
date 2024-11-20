def _get_graph(model):
    try:
        return _get_script_mod(model)._c.dump_to_str(True, False, False)
    except AttributeError:
        return model.graph.str()
