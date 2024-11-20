import pandas as pd
import pytest
import torch
import torch.nn as nn

from deeplite_torch_zoo.utils.profiler import profile_ram
from deeplite_torch_zoo.utils.torch_utils import TORCH_2_0
from deeplite_torch_zoo import get_model

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.mark.parametrize(
    (
        'columns',
        'num_nodes',
        'peak_ram',
        'input_shape_0',
        'scope_1',
        'output_shape_3',
    ),
    [
        (
            ['weight', 'bias', 'input_shape', 'output_shape', 'in_tensors',
             'out_tensors', 'active_blocks', 'ram', 'scope'],
            8,
            262144,
            [1, 3, 32, 32],
            'relu1',
            [1, 32, 32, 32],
        )
    ]
)


@pytest.mark.skipif(not TORCH_2_0, reason='scope tracing not supported on torch<2.0')


@pytest.mark.skipif(not TORCH_2_0, reason='scope tracing not supported on torch<2.0')


class DummySequentialForModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.Conv2d(16, 2, 1)
        )

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x


class SequentialSubclass(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, x):
        for m in self:
            x = m(x)
        return x


class DummySequentialSequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers1 = SequentialSubclass(
            nn.Conv2d(3, 16, 1),
            nn.Conv2d(16,16, 1)
        )
        self.all_layers = nn.Sequential(
            layers1,
        )

    def forward(self, x):
        return self.all_layers(x)


class DummySequentialSequentialForModel(nn.Module):
    def __init__(self):
        super().__init__()
        layers1 = nn.Sequential(
            nn.Conv2d(3, 16, 1),
            nn.Conv2d(16, 16, 1)
        )
        self.all_layers = nn.Sequential(
            layers1
        )

    def forward(self, x):
        for m in self.all_layers:
            x = m(x)
        return x


class EdgeCaseSequentialModel(nn.Module):
    def __init__(self):
        super().__init__()
        # parent of sequential has same name as sequential 'all_layers'
        self.all_layers = DummySequentialSequentialForModel()

    def forward(self, x):
        return self.all_layers(x)





@pytest.mark.parametrize(
    (
        'columns',
        'num_nodes',
        'peak_ram',
        'input_shape_0',
        'scope_1',
        'output_shape_3',
    ),
    [
        (
            ['weight', 'bias', 'input_shape', 'output_shape', 'in_tensors',
             'out_tensors', 'active_blocks', 'ram', 'scope'],
            8,
            262144,
            [1, 3, 32, 32],
            'relu1',
            [1, 32, 32, 32],
        )
    ]
)
def test_ram_profiling(
        columns,
        num_nodes,
        peak_ram,
        input_shape_0,
        scope_1,
        output_shape_3,
    ):
    model = DummyModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    nodes_info = profile_ram(model, input_tensor, detailed=True)

    assert isinstance(nodes_info, pd.DataFrame)
    assert all(column in nodes_info.columns for column in columns)
    assert len(nodes_info) == num_nodes
    assert nodes_info.ram.max() == peak_ram
    assert nodes_info.iloc[0].input_shape[0] == input_shape_0
    assert nodes_info.iloc[1].scope == scope_1
    assert nodes_info.iloc[3].output_shape[0] == output_shape_3


@pytest.mark.skipif(not TORCH_2_0, reason='scope tracing not supported on torch<2.0')
def test_ram_scope():
    model = DummySequentialForModel()
    input_tensor = torch.randn(1, 3, 32, 32)
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert nodes_info.iloc[0].scope == 'layers.0'
    assert get_submodule(model, nodes_info.iloc[0].scope)

    model = DummySequentialSequentialModel()
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert nodes_info.iloc[0].scope == 'all_layers.0.0'
    assert get_submodule(model, nodes_info.iloc[0].scope)

    model = DummySequentialSequentialForModel()
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert nodes_info.iloc[0].scope == 'all_layers.0.0'
    assert get_submodule(model, nodes_info.iloc[0].scope)

    model = nn.Sequential(
        nn.Conv2d(3, 16, 1),
        nn.Conv2d(16, 2, 1)
    )
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert nodes_info.iloc[0].scope == '0'
    assert get_submodule(model, nodes_info.iloc[0].scope)

    # NOTE Known fail case where parent.parent.name == parent.name
    # current scope = A.B
    # total scope = A.A.B
    # Second A is sequential called with for loop
    model = EdgeCaseSequentialModel()
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    with pytest.raises(AssertionError):
        assert nodes_info.iloc[0].scope == 'all_layers.all_layers.0.0'


@pytest.mark.skipif(not TORCH_2_0, reason='scope tracing not supported on torch<2.0')
def test_ram_scope_yolo():
    model = get_model('yolo5n', 'coco', False)
    input_tensor = torch.randn(1, 3, 96, 96)
    nodes_info = profile_ram(model, input_tensor, detailed=True)
    assert sum(nodes_info['scope'] == 'model.23.cv1.conv') == 1


class DummySequentialForModel(nn.Module):



class SequentialSubclass(nn.Sequential):



class DummySequentialSequentialModel(nn.Module):



class DummySequentialSequentialForModel(nn.Module):



class EdgeCaseSequentialModel(nn.Module):



def get_submodule(module, submodule_name: str) -> nn.Module:
    if submodule_name == "":
        return module

    err_msg = 'Cannot retrieve submodule {target} from the model: '

    for submodule in  submodule_name.split("."):
        if not hasattr(module, submodule):
            err_msg += module._get_name() + \
                " has no " "attribute `" + submodule + "`"  # pylint: disable=protected-access
            raise AttributeError(err_msg)

        module = getattr(module, submodule)

        if not isinstance(module, nn.Module):
            err_msg += "`" + submodule + "` is not " "an nn.Module"
            raise AttributeError(err_msg)

    return module