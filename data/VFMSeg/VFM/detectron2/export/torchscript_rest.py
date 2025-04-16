# Copyright (c) Facebook, Inc. and its affiliates.

import os
import torch

from detectron2.utils.file_io import PathManager

from .torchscript_patch import freeze_training_mode, patch_instances

__all__ = ["scripting_with_instances", "dump_torchscript_IR"]




# alias for old name
export_torchscript_with_instances = scripting_with_instances



    # Dump pretty-printed code: https://pytorch.org/docs/stable/jit.html#inspecting-code
    with PathManager.open(os.path.join(dir, "model_ts_code.txt"), "w") as f:

        def get_code(mod):
            # Try a few ways to get code using private attributes.
            try:
                # This contains more information than just `mod.code`
                return _get_script_mod(mod)._c.code
            except AttributeError:
                pass
            try:
                return mod.code
            except AttributeError:
                return None

        def dump_code(prefix, mod):
            code = get_code(mod)
            name = prefix or "root model"
            if code is None:
                f.write(f"Could not found code for {name} (type={mod.original_name})\n")
                f.write("\n")
            else:
                f.write(f"\nCode for {name}, type={mod.original_name}:\n")
                f.write(code)
                f.write("\n")
                f.write("-" * 80)

            for name, m in mod.named_children():
                dump_code(prefix + "." + name, m)

        if isinstance(model, torch.jit.ScriptFunction):
            f.write(get_code(model))
        else:
            dump_code("", model)



        if isinstance(model, torch.jit.ScriptFunction):
            f.write(get_code(model))
        else:
            dump_code("", model)

    def _get_graph(model):
        try:
            # Recursively dump IR of all modules
            return _get_script_mod(model)._c.dump_to_str(True, False, False)
        except AttributeError:
            return model.graph.str()

    with PathManager.open(os.path.join(dir, "model_ts_IR.txt"), "w") as f:
        f.write(_get_graph(model))

    # Dump IR of the entire graph (all submodules inlined)
    with PathManager.open(os.path.join(dir, "model_ts_IR_inlined.txt"), "w") as f:
        f.write(str(model.inlined_graph))

    if not isinstance(model, torch.jit.ScriptFunction):
        # Dump the model structure in pytorch style
        with PathManager.open(os.path.join(dir, "model.txt"), "w") as f:
            f.write(str(model))