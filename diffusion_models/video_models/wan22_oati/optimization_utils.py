"""
"""
import contextlib
from contextvars import ContextVar
from io import BytesIO
from typing import Any
from typing import cast
from unittest.mock import patch

import torch
from torch._inductor.package.package import package_aoti
from torch.export.pt2_archive._package import AOTICompiledModel
from torch.export.pt2_archive._package_weights import Weights


INDUCTOR_CONFIGS_OVERRIDES = {
    'aot_inductor.package_constants_in_so': False,
    'aot_inductor.package_constants_on_disk': True,
    'aot_inductor.package': True,
}


class ZeroGPUWeights:
    def __init__(self, constants_map: dict[str, torch.Tensor], to_cuda: bool = False):
        if to_cuda:
            self.constants_map = {name: tensor.to('cuda') for name, tensor in constants_map.items()}
        else:
            self.constants_map = constants_map
    def __reduce__(self):
        constants_map: dict[str, torch.Tensor] = {}
        for name, tensor in self.constants_map.items():
            tensor_ = torch.empty_like(tensor, device='cpu').pin_memory()
            constants_map[name] = tensor_.copy_(tensor).detach().share_memory_()
        return ZeroGPUWeights, (constants_map, True)


class ZeroGPUCompiledModel:
    def __init__(self, archive_file: torch.types.FileLike, weights: ZeroGPUWeights):
        self.archive_file = archive_file
        self.weights = weights
        self.compiled_model: ContextVar[AOTICompiledModel | None] = ContextVar('compiled_model', default=None)
    def __call__(self, *args, **kwargs):
        if (compiled_model := self.compiled_model.get()) is None:
            compiled_model = cast(AOTICompiledModel, torch._inductor.aoti_load_package(self.archive_file))
            compiled_model.load_constants(self.weights.constants_map, check_full_update=True, user_managed=True)
            self.compiled_model.set(compiled_model)
        return compiled_model(*args, **kwargs)
    def __reduce__(self):
        return ZeroGPUCompiledModel, (self.archive_file, self.weights)


def aoti_compile(
    exported_program: torch.export.ExportedProgram,
    inductor_configs: dict[str, Any] | None = None,
):
    inductor_configs = (inductor_configs or {}) | INDUCTOR_CONFIGS_OVERRIDES
    gm = cast(torch.fx.GraphModule, exported_program.module())
    assert exported_program.example_inputs is not None
    args, kwargs = exported_program.example_inputs
    artifacts = torch._inductor.aot_compile(gm, args, kwargs, options=inductor_configs)
    archive_file = BytesIO()
    files: list[str | Weights] = [file for file in artifacts if isinstance(file, str)]
    package_aoti(archive_file, files)
    weights, = (artifact for artifact in artifacts if isinstance(artifact, Weights))
    zerogpu_weights = ZeroGPUWeights({name: weights.get_weight(name)[0] for name in weights})
    return ZeroGPUCompiledModel(archive_file, zerogpu_weights)


@contextlib.contextmanager
def capture_component_call(
    pipeline: Any,
    component_name: str,
    component_method='forward',
):

    class CapturedCallException(Exception):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args = args
            self.kwargs = kwargs

    class CapturedCall:
        def __init__(self):
            self.args: tuple[Any, ...] = ()
            self.kwargs: dict[str, Any] = {}

    component = getattr(pipeline, component_name)
    captured_call = CapturedCall()

    def capture_call(*args, **kwargs):
        raise CapturedCallException(*args, **kwargs)

    with patch.object(component, component_method, new=capture_call):
        try:
            yield captured_call
        except CapturedCallException as e:
            captured_call.args = e.args
            captured_call.kwargs = e.kwargs


def drain_module_parameters(module: torch.nn.Module):
    state_dict_meta = {name: {'device': tensor.device, 'dtype': tensor.dtype} for name, tensor in module.state_dict().items()}
    state_dict = {name: torch.nn.Parameter(torch.empty_like(tensor, device='cpu')) for name, tensor in module.state_dict().items()}
    module.load_state_dict(state_dict, assign=True)
    for name, param in state_dict.items():
        meta = state_dict_meta[name]
        param.data = torch.Tensor([]).to(**meta)