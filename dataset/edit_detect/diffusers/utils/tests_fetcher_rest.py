# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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

"""
Welcome to tests_fetcher V2.

This util is designed to fetch tests to run on a PR so that only the tests impacted by the modifications are run, and
when too many models are being impacted, only run the tests of a subset of core models. It works like this.

Stage 1: Identify the modified files. For jobs that run on the main branch, it's just the diff with the last commit.
On a PR, this takes all the files from the branching point to the current commit (so all modifications in a PR, not
just the last commit) but excludes modifications that are on docstrings or comments only.

Stage 2: Extract the tests to run. This is done by looking at the imports in each module and test file: if module A
imports module B, then changing module B impacts module A, so the tests using module A should be run. We thus get the
dependencies of each model and then recursively builds the 'reverse' map of dependencies to get all modules and tests
impacted by a given file. We then only keep the tests (and only the core models tests if there are too many modules).

Caveats:
  - This module only filters tests by files (not individual tests) so it's better to have tests for different things
    in different files.
  - This module assumes inits are just importing things, not really building objects, so it's better to structure
    them this way and move objects building in separate submodules.

Usage:

Base use to fetch the tests in a pull request

```bash
python utils/tests_fetcher.py
```

Base use to fetch the tests on a the main branch (with diff from the last commit):

```bash
python utils/tests_fetcher.py --diff_with_last_commit
```
"""

import argparse
import collections
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from git import Repo


PATH_TO_REPO = Path(__file__).parent.parent.resolve()
PATH_TO_EXAMPLES = PATH_TO_REPO / "examples"
PATH_TO_DIFFUSERS = PATH_TO_REPO / "src/diffusers"
PATH_TO_TESTS = PATH_TO_REPO / "tests"

# Ignore fixtures in tests folder
# Ignore lora since they are always tested
MODULES_TO_IGNORE = ["fixtures", "lora"]

IMPORTANT_PIPELINES = [
    "controlnet",
    "stable_diffusion",
    "stable_diffusion_2",
    "stable_diffusion_xl",
    "stable_video_diffusion",
    "deepfloyd_if",
    "kandinsky",
    "kandinsky2_2",
    "text_to_video_synthesis",
    "wuerstchen",
]


@contextmanager
























# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+(\.+\S+)\s+import\s+([^\n]+) -> Line only contains from .xxx import yyy and we catch .xxx and yyy
# (?=\n) -> Look-ahead to a new line. We can't just put \n here or using find_all on this re will only catch every
#           other import.
_re_single_line_relative_imports = re.compile(r"(?:^|\n)\s*from\s+(\.+\S+)\s+import\s+([^\n]+)(?=\n)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+(\.+\S+)\s+import\s+\(([^\)]+)\) -> Line continues with from .xxx import (yyy) and we catch .xxx and yyy
# yyy will take multiple lines otherwise there wouldn't be parenthesis.
_re_multi_line_relative_imports = re.compile(r"(?:^|\n)\s*from\s+(\.+\S+)\s+import\s+\(([^\)]+)\)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+transformers(\S*)\s+import\s+([^\n]+) -> Line only contains from transformers.xxx import yyy and we catch
#           .xxx and yyy
# (?=\n) -> Look-ahead to a new line. We can't just put \n here or using find_all on this re will only catch every
#           other import.
_re_single_line_direct_imports = re.compile(r"(?:^|\n)\s*from\s+diffusers(\S*)\s+import\s+([^\n]+)(?=\n)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+transformers(\S*)\s+import\s+\(([^\)]+)\) -> Line continues with from transformers.xxx import (yyy) and we
# catch .xxx and yyy. yyy will take multiple lines otherwise there wouldn't be parenthesis.
_re_multi_line_direct_imports = re.compile(r"(?:^|\n)\s*from\s+diffusers(\S*)\s+import\s+\(([^\)]+)\)")































    # Build the test map
    test_map = {module: [f for f in deps if is_test(f)] for module, deps in reverse_map.items()}

    return test_map


def check_imports_all_exist():
    """
    Isn't used per se by the test fetcher but might be used later as a quality check. Putting this here for now so the
    code is not lost. This checks all imports in a given file do exist.
    """
    cache = {}
    all_modules = list(PATH_TO_DIFFUSERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py"))
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    direct_deps = {m: get_module_dependencies(m, cache=cache) for m in all_modules}

    for module, deps in direct_deps.items():
        for dep in deps:
            if not (PATH_TO_REPO / dep).is_file():
                print(f"{module} has dependency on {dep} which does not exist.")


def _print_list(l) -> str:
    """
    Pretty print a list of elements with one line per element and a - starting each line.
    """
    return "\n".join([f"- {f}" for f in l])


def update_test_map_with_core_pipelines(json_output_file: str):
    print(f"\n### ADD CORE PIPELINE TESTS ###\n{_print_list(IMPORTANT_PIPELINES)}")
    with open(json_output_file, "rb") as fp:
        test_map = json.load(fp)

    # Add core pipelines as their own test group
    test_map["core_pipelines"] = " ".join(
        sorted([str(PATH_TO_TESTS / f"pipelines/{pipe}") for pipe in IMPORTANT_PIPELINES])
    )

    # If there are no existing pipeline tests save the map
    if "pipelines" not in test_map:
        with open(json_output_file, "w", encoding="UTF-8") as fp:
            json.dump(test_map, fp, ensure_ascii=False)

    pipeline_tests = test_map.pop("pipelines")
    pipeline_tests = pipeline_tests.split(" ")

    # Remove core pipeline tests from the fetched pipeline tests
    updated_pipeline_tests = []
    for pipe in pipeline_tests:
        if pipe == "tests/pipelines" or Path(pipe).parts[2] in IMPORTANT_PIPELINES:
            continue
        updated_pipeline_tests.append(pipe)

    if len(updated_pipeline_tests) > 0:
        test_map["pipelines"] = " ".join(sorted(updated_pipeline_tests))

    with open(json_output_file, "w", encoding="UTF-8") as fp:
        json.dump(test_map, fp, ensure_ascii=False)


def create_json_map(test_files_to_run: List[str], json_output_file: Optional[str] = None):
    """
    Creates a map from a list of tests to run to easily split them by category, when running parallelism of slow tests.

    Args:
        test_files_to_run (`List[str]`): The list of tests to run.
        json_output_file (`str`): The path where to store the built json map.
    """
    if json_output_file is None:
        return

    test_map = {}
    for test_file in test_files_to_run:
        # `test_file` is a path to a test folder/file, starting with `tests/`. For example,
        #   - `tests/models/bert/test_modeling_bert.py` or `tests/models/bert`
        #   - `tests/trainer/test_trainer.py` or `tests/trainer`
        #   - `tests/test_modeling_common.py`
        names = test_file.split(os.path.sep)
        module = names[1]
        if module in MODULES_TO_IGNORE:
            continue

        if len(names) > 2 or not test_file.endswith(".py"):
            # test folders under `tests` or python files under them
            # take the part like tokenization, `pipeline`, etc. for other test categories
            key = os.path.sep.join(names[1:2])
        else:
            # common test files directly under `tests/`
            key = "common"

        if key not in test_map:
            test_map[key] = []
        test_map[key].append(test_file)

    # sort the keys & values
    keys = sorted(test_map.keys())
    test_map = {k: " ".join(sorted(test_map[k])) for k in keys}

    with open(json_output_file, "w", encoding="UTF-8") as fp:
        json.dump(test_map, fp, ensure_ascii=False)


def infer_tests_to_run(
    output_file: str,
    diff_with_last_commit: bool = False,
    json_output_file: Optional[str] = None,
):
    """
    The main function called by the test fetcher. Determines the tests to run from the diff.

    Args:
        output_file (`str`):
            The path where to store the summary of the test fetcher analysis. Other files will be stored in the same
            folder:

            - examples_test_list.txt: The list of examples tests to run.
            - test_repo_utils.txt: Will indicate if the repo utils tests should be run or not.
            - doctest_list.txt: The list of doctests to run.

        diff_with_last_commit (`bool`, *optional*, defaults to `False`):
            Whether to analyze the diff with the last commit (for use on the main branch after a PR is merged) or with
            the branching point from main (for use on each PR).
        filter_models (`bool`, *optional*, defaults to `True`):
            Whether or not to filter the tests to core models only, when a file modified results in a lot of model
            tests.
        json_output_file (`str`, *optional*):
            The path where to store the json file mapping categories of tests to tests to run (used for parallelism or
            the slow tests).
    """
    modified_files = get_modified_python_files(diff_with_last_commit=diff_with_last_commit)
    print(f"\n### MODIFIED FILES ###\n{_print_list(modified_files)}")
    # Create the map that will give us all impacted modules.
    reverse_map = create_reverse_dependency_map()
    impacted_files = modified_files.copy()
    for f in modified_files:
        if f in reverse_map:
            impacted_files.extend(reverse_map[f])

    # Remove duplicates
    impacted_files = sorted(set(impacted_files))
    print(f"\n### IMPACTED FILES ###\n{_print_list(impacted_files)}")

    # Grab the corresponding test files:
    if any(x in modified_files for x in ["setup.py"]):
        test_files_to_run = ["tests", "examples"]

    # in order to trigger pipeline tests even if no code change at all
    if "tests/utils/tiny_model_summary.json" in modified_files:
        test_files_to_run = ["tests"]
        any(f.split(os.path.sep)[0] == "utils" for f in modified_files)
    else:
        # All modified tests need to be run.
        test_files_to_run = [
            f for f in modified_files if f.startswith("tests") and f.split(os.path.sep)[-1].startswith("test")
        ]
        # Then we grab the corresponding test files.
        test_map = create_module_to_test_map(reverse_map=reverse_map)
        for f in modified_files:
            if f in test_map:
                test_files_to_run.extend(test_map[f])
        test_files_to_run = sorted(set(test_files_to_run))
        # Make sure we did not end up with a test file that was removed
        test_files_to_run = [f for f in test_files_to_run if (PATH_TO_REPO / f).exists()]

        any(f.split(os.path.sep)[0] == "utils" for f in modified_files)

    examples_tests_to_run = [f for f in test_files_to_run if f.startswith("examples")]
    test_files_to_run = [f for f in test_files_to_run if not f.startswith("examples")]
    print(f"\n### TEST TO RUN ###\n{_print_list(test_files_to_run)}")
    if len(test_files_to_run) > 0:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(" ".join(test_files_to_run))

        # Create a map that maps test categories to test files, i.e. `models/bert` -> [...test_modeling_bert.py, ...]

        # Get all test directories (and some common test files) under `tests` and `tests/models` if `test_files_to_run`
        # contains `tests` (i.e. when `setup.py` is changed).
        if "tests" in test_files_to_run:
            test_files_to_run = get_all_tests()

        create_json_map(test_files_to_run, json_output_file)

    print(f"\n### EXAMPLES TEST TO RUN ###\n{_print_list(examples_tests_to_run)}")
    if len(examples_tests_to_run) > 0:
        # We use `all` in the case `commit_flags["test_all"]` as well as in `create_circleci_config.py` for processing
        if examples_tests_to_run == ["examples"]:
            examples_tests_to_run = ["all"]
        example_file = Path(output_file).parent / "examples_test_list.txt"
        with open(example_file, "w", encoding="utf-8") as f:
            f.write(" ".join(examples_tests_to_run))


def filter_tests(output_file: str, filters: List[str]):
    """
    Reads the content of the output file and filters out all the tests in a list of given folders.

    Args:
        output_file (`str` or `os.PathLike`): The path to the output file of the tests fetcher.
        filters (`List[str]`): A list of folders to filter.
    """
    if not os.path.isfile(output_file):
        print("No test file found.")
        return
    with open(output_file, "r", encoding="utf-8") as f:
        test_files = f.read().split(" ")

    if len(test_files) == 0 or test_files == [""]:
        print("No tests to filter.")
        return

    if test_files == ["tests"]:
        test_files = [os.path.join("tests", f) for f in os.listdir("tests") if f not in ["__init__.py"] + filters]
    else:
        test_files = [f for f in test_files if f.split(os.path.sep)[1] not in filters]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(" ".join(test_files))


def parse_commit_message(commit_message: str) -> Dict[str, bool]:
    """
    Parses the commit message to detect if a command is there to skip, force all or part of the CI.

    Args:
        commit_message (`str`): The commit message of the current commit.

    Returns:
        `Dict[str, bool]`: A dictionary of strings to bools with keys the following keys: `"skip"`,
        `"test_all_models"` and `"test_all"`.
    """
    if commit_message is None:
        return {"skip": False, "no_filter": False, "test_all": False}

    command_search = re.search(r"\[([^\]]*)\]", commit_message)
    if command_search is not None:
        command = command_search.groups()[0]
        command = command.lower().replace("-", " ").replace("_", " ")
        skip = command in ["ci skip", "skip ci", "circleci skip", "skip circleci"]
        no_filter = set(command.split(" ")) == {"no", "filter"}
        test_all = set(command.split(" ")) == {"test", "all"}
        return {"skip": skip, "no_filter": no_filter, "test_all": test_all}
    else:
        return {"skip": False, "no_filter": False, "test_all": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="test_list.txt", help="Where to store the list of tests to run"
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        default="test_map.json",
        help="Where to store the tests to run in a dictionary format mapping test categories to test files",
    )
    parser.add_argument(
        "--diff_with_last_commit",
        action="store_true",
        help="To fetch the tests between the current commit and the last commit",
    )
    parser.add_argument(
        "--filter_tests",
        action="store_true",
        help="Will filter the pipeline/repo utils tests outside of the generated list of tests.",
    )
    parser.add_argument(
        "--print_dependencies_of",
        type=str,
        help="Will only print the tree of modules depending on the file passed.",
        default=None,
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        help="The commit message (which could contain a command to force all tests or skip the CI).",
        default=None,
    )
    args = parser.parse_args()
    if args.print_dependencies_of is not None:
        print_tree_deps_of(args.print_dependencies_of)
    else:
        repo = Repo(PATH_TO_REPO)
        commit_message = repo.head.commit.message
        commit_flags = parse_commit_message(commit_message)
        if commit_flags["skip"]:
            print("Force-skipping the CI")
            quit()
        if commit_flags["no_filter"]:
            print("Running all tests fetched without filtering.")
        if commit_flags["test_all"]:
            print("Force-launching all tests")

        diff_with_last_commit = args.diff_with_last_commit
        if not diff_with_last_commit and not repo.head.is_detached and repo.head.ref == repo.refs.main:
            print("main branch detected, fetching tests against last commit.")
            diff_with_last_commit = True

        if not commit_flags["test_all"]:
            try:
                infer_tests_to_run(
                    args.output_file,
                    diff_with_last_commit=diff_with_last_commit,
                    json_output_file=args.json_output_file,
                )
                filter_tests(args.output_file, ["repo_utils"])
                update_test_map_with_core_pipelines(json_output_file=args.json_output_file)

            except Exception as e:
                print(f"\nError when trying to grab the relevant tests: {e}\n\nRunning all tests.")
                commit_flags["test_all"] = True

        if commit_flags["test_all"]:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write("tests")
            example_file = Path(args.output_file).parent / "examples_test_list.txt"
            with open(example_file, "w", encoding="utf-8") as f:
                f.write("all")

            test_files_to_run = get_all_tests()
            create_json_map(test_files_to_run, args.json_output_file)
            update_test_map_with_core_pipelines(json_output_file=args.json_output_file)