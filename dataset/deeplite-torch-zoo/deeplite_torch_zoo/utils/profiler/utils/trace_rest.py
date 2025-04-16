import re
import warnings

import torch
import torch.jit

from deeplite_torch_zoo.utils.profiler.utils.flatten import Flatten
from deeplite_torch_zoo.utils.profiler.utils.ir import Graph, Node, Variable
from deeplite_torch_zoo.utils.torch_utils import TORCH_2_0


class ScopeNameContextManager:
    """
    A context manager to handle scope names in PyTorch model tracing.
    This class temporarily overrides the '_slow_forward' method of torch.nn.Module
    to capture scope names accurately during the tracing process.
    """

    def __init__(self):
        self.original_slow_forward = None

    def __enter__(self):
        self.original_slow_forward = torch.nn.Module._slow_forward
        torch.nn.Module._slow_forward = self._patched_slow_forward

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.nn.Module._slow_forward = self.original_slow_forward

    @staticmethod
    def _patched_slow_forward(module, *inputs, **kwargs):
        tracing_state = torch._C._get_tracing_state()

        if not tracing_state or isinstance(module.forward, torch._C.ScriptMethod):
            return module.forward(*inputs, **kwargs)

        if not hasattr(tracing_state, '_traced_module_stack'):
            tracing_state._traced_module_stack = []

        module_name = ScopeNameContextManager._get_tracing_name(module, tracing_state)
        scope_name = f'{module._get_name()}[{module_name}]' if module_name else module._get_name()
        tracing_state.push_scope(scope_name)
        tracing_state._traced_module_stack.append(module)

        try:
            result = module.forward(*inputs, **kwargs)
        finally:
            tracing_state.pop_scope()
            tracing_state._traced_module_stack.pop()

        return result

    @staticmethod
    def _get_tracing_name(module, tracing_state):
        if not tracing_state._traced_module_stack:
            return None

        parent_module = tracing_state._traced_module_stack[-1]
        for name, child in parent_module.named_children():
            if child is module:
                return name
        return None









    @staticmethod

    @staticmethod


def filter_torch_scope(node):
    """
    Extracts and formats the module name from a torch.Node traced scope name
    Relies on torchscript-based onnx trace: torch.onnx.trace

    Input: torch.Node node
    Output: str scope

    The node scope collected by torch.onnx.trace is string containing a
    '/' separated list of ModuleType::module_name
    The purpose of this function is to extract the module_names from this list
    Named submodules will appear in the list as expected, i.e. Conv2d[conv1]
    Unnamed submodules in a Sequential module or ModuleList will not always appear as expected
    This is a known issue https://github.com/pytorch/pytorch/issues/90439
    if the Sequential module's forward function is called:
        node.scope = Model::/Sequential::layers/Conv2d::layers.0
        Sequential module is present in scope list, so remove repeated 'layers' name
    if the submodule is called directly: (for m in sequential: m(x))
        node.scope = Model::/Conv2d::layers.0
        sequential module is not present in scope list, so preserve 'layers' name

    Future work:
    switch to torch.fx or torch dynamo tracing since torchscript tracing is not actively supported
    """
    if not TORCH_2_0:
        return filter_torch_scope_torch12(node)

    scope = node.scopeName()
    if scope == '':
        return scope
    module_pairs = scope.split('/')
    if len(module_pairs) == 1:
        return module_pairs[0]

    module_names = []
    module_types = []
    for module_pair in module_pairs:
        type_name = module_pair.split('::')
        if len(type_name) == 1:
            # no name provided, just use model type
            # module_names.append(type_name[0])
            continue

        module_type, module_name = type_name[0], type_name[-1]
        module_name_split = module_name.split('.')
        if len(module_name_split) == 1:
            # base case, where module is a named submodule
            module_names.append(module_name)
        else:
            # Edge case: we are in child of sequential, so module name includes parent name
            # to avoid repetition of parent name in scope, check if name is already on stack
            # NOTE: there exists an unsupported edge case, see test_ram_profiler.py for info
            names_to_add = [module_name_split[-1]]
            types_to_add = [module_type]
            module_names_idx = len(module_names) - 1
            split_idx = len(module_name_split) - 2
            last_saved_name = module_names[module_names_idx]

            # from parent of module, check if parent in the stack, if not add it to names to add
            # handles nested sequentials of any depth
            for split_idx in range(len(module_name_split) - 2, -1, -1):
                parent_name = module_name_split[split_idx]
                if parent_name == last_saved_name:
                    break
                names_to_add.append(parent_name)
                types_to_add.append('__hidden_sequential__')  # for debug only
            module_names.extend(names_to_add[::-1])
            module_types.extend(types_to_add[::-1])

    # Remove first name in scope (model name: '')
    filtered_scope = '.'.join(module_names[1:])
    return filtered_scope


def filter_torch_scope_torch12(node):
    """
    Extracts and formats the module name from a PyTorch graph node's scope name.
    """
    scope = node.scopeName().replace('Flatten/', '', 1).replace('Flatten', '', 1)
    scope_list = re.findall(r"\[.*?\]", scope)

    module_name = ''
    if len(scope_list) >= 2:
        module_name = '.'.join(token.strip('[]') for token in scope_list[1:])

    return module_name


def trace(model, args=(), kwargs=None):
    assert kwargs is None, (
        'Keyword arguments are not supported for now. '
        'Please use positional arguments instead!'
    )

    # initialize module scope caching, supported as of torch 1.13
    if TORCH_2_0:
        torch.onnx.utils._setup_trace_module_map(model, False)
        graph, _ = torch.jit._get_trace_graph(Flatten(model), args, kwargs)
    else:
        with warnings.catch_warnings(record=True), ScopeNameContextManager():
            graph, _ = torch.jit._get_trace_graph(Flatten(model), args, kwargs)

    variables = {}
    for x in graph.nodes():
        for v in list(x.inputs()) + list(x.outputs()):
            if 'tensor' in v.type().kind().lower():
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=v.type().scalarType(),
                    shape=v.type().sizes(),
                )
            else:
                variables[v] = Variable(
                    name=v.debugName(),
                    dtype=str(v.type()),
                )

    nodes = []
    for x in graph.nodes():
        scope = filter_torch_scope(x)
        node = Node(
            operator=x.kind(),
            attributes={s: getattr(x, x.kindOf(s))(s) for s in x.attributeNames()},
            inputs=[variables[v] for v in x.inputs() if v in variables],
            outputs=[variables[v] for v in x.outputs() if v in variables],
            scope=scope,
        )
        nodes.append(node)

    graph = Graph(
        name=model.__class__.__module__ + '.' + model.__class__.__name__,
        variables=list(variables.values()),
        inputs=[variables[v] for v in graph.inputs() if v in variables],
        outputs=[variables[v] for v in graph.outputs() if v in variables],
        nodes=nodes,
    )

    return graph