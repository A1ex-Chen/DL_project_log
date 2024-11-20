# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import importlib.util
import os
import re
from pathlib import Path


PATH_TO_TRANSFORMERS = "src/transformers"


# Matches is_xxx_available()
_re_backend = re.compile(r"is\_([a-z_]*)_available()")
# Catches a one-line _import_struct = {xxx}
_re_one_line_import_struct = re.compile(r"^_import_structure\s+=\s+\{([^\}]+)\}")
# Catches a line with a key-values pattern: "bla": ["foo", "bar"]
_re_import_struct_key_value = re.compile(r'\s+"\S*":\s+\[([^\]]*)\]')
# Catches a line if not is_foo_available
_re_test_backend = re.compile(r"^\s*if\s+not\s+is\_[a-z_]*\_available\(\)")
# Catches a line _import_struct["bla"].append("foo")
_re_import_struct_add_one = re.compile(r'^\s*_import_structure\["\S*"\]\.append\("(\S*)"\)')
# Catches a line _import_struct["bla"].extend(["foo", "bar"]) or _import_struct["bla"] = ["foo", "bar"]
_re_import_struct_add_many = re.compile(r"^\s*_import_structure\[\S*\](?:\.extend\(|\s*=\s+)\[([^\]]*)\]")
# Catches a line with an object between quotes and a comma:     "MyModel",
_re_quote_object = re.compile(r'^\s+"([^"]+)",')
# Catches a line with objects between brackets only:    ["foo", "bar"],
_re_between_brackets = re.compile(r"^\s+\[([^\]]+)\]")
# Catches a line with from foo import bar, bla, boo
_re_import = re.compile(r"\s+from\s+\S*\s+import\s+([^\(\s].*)\n")
# Catches a line with try:
_re_try = re.compile(r"^\s*try:")
# Catches a line with else:
_re_else = re.compile(r"^\s*else:")












IGNORE_SUBMODULES = [
    "convert_pytorch_checkpoint_to_tf2",
    "modeling_flax_pytorch_utils",
]



    if list(import_dict_objects.keys()) != list(type_hint_objects.keys()):
        return ["Both sides of the init do not have the same backends!"]

    errors = []
    for key in import_dict_objects.keys():
        duplicate_imports = find_duplicates(import_dict_objects[key])
        if duplicate_imports:
            errors.append(f"Duplicate _import_structure definitions for: {duplicate_imports}")
        duplicate_type_hints = find_duplicates(type_hint_objects[key])
        if duplicate_type_hints:
            errors.append(f"Duplicate TYPE_CHECKING objects for: {duplicate_type_hints}")

        if sorted(set(import_dict_objects[key])) != sorted(set(type_hint_objects[key])):
            name = "base imports" if key == "none" else f"{key} backend"
            errors.append(f"Differences for {name}:")
            for a in type_hint_objects[key]:
                if a not in import_dict_objects[key]:
                    errors.append(f"  {a} in TYPE_HINT but not in _import_structure.")
            for a in import_dict_objects[key]:
                if a not in type_hint_objects[key]:
                    errors.append(f"  {a} in _import_structure but not in TYPE_HINT.")
    return errors


def check_all_inits():
    """
    Check all inits in the transformers repo and raise an error if at least one does not define the same objects in
    both halves.
    """
    failures = []
    for root, _, files in os.walk(PATH_TO_TRANSFORMERS):
        if "__init__.py" in files:
            fname = os.path.join(root, "__init__.py")
            objects = parse_init(fname)
            if objects is not None:
                errors = analyze_results(*objects)
                if len(errors) > 0:
                    errors[0] = f"Problem in {fname}, both halves do not define the same objects.\n{errors[0]}"
                    failures.append("\n".join(errors))
    if len(failures) > 0:
        raise ValueError("\n\n".join(failures))


def get_transformers_submodules():
    """
    Returns the list of Transformers submodules.
    """
    submodules = []
    for path, directories, files in os.walk(PATH_TO_TRANSFORMERS):
        for folder in directories:
            # Ignore private modules
            if folder.startswith("_"):
                directories.remove(folder)
                continue
            # Ignore leftovers from branches (empty folders apart from pycache)
            if len(list((Path(path) / folder).glob("*.py"))) == 0:
                continue
            short_path = str((Path(path) / folder).relative_to(PATH_TO_TRANSFORMERS))
            submodule = short_path.replace(os.path.sep, ".")
            submodules.append(submodule)
        for fname in files:
            if fname == "__init__.py":
                continue
            short_path = str((Path(path) / fname).relative_to(PATH_TO_TRANSFORMERS))
            submodule = short_path.replace(".py", "").replace(os.path.sep, ".")
            if len(submodule.split(".")) == 1:
                submodules.append(submodule)
    return submodules


IGNORE_SUBMODULES = [
    "convert_pytorch_checkpoint_to_tf2",
    "modeling_flax_pytorch_utils",
]


def check_submodules():
    # This is to make sure the transformers module imported is the one in the repo.
    spec = importlib.util.spec_from_file_location(
        "transformers",
        os.path.join(PATH_TO_TRANSFORMERS, "__init__.py"),
        submodule_search_locations=[PATH_TO_TRANSFORMERS],
    )
    transformers = spec.loader.load_module()

    module_not_registered = [
        module
        for module in get_transformers_submodules()
        if module not in IGNORE_SUBMODULES and module not in transformers._import_structure.keys()
    ]
    if len(module_not_registered) > 0:
        list_of_modules = "\n".join(f"- {module}" for module in module_not_registered)
        raise ValueError(
            "The following submodules are not properly registered in the main init of Transformers:\n"
            f"{list_of_modules}\n"
            "Make sure they appear somewhere in the keys of `_import_structure` with an empty list as value."
        )


if __name__ == "__main__":
    check_all_inits()
    check_submodules()