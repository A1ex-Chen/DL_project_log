from typing import TYPE_CHECKING

from ..utils import DIFFUSERS_SLOW_IMPORT, _LazyModule, deprecate
from ..utils.import_utils import is_peft_available, is_torch_available, is_transformers_available




if is_transformers_available():



_import_structure = {}

if is_torch_available():
    _import_structure["single_file_model"] = ["FromOriginalModelMixin"]
    _import_structure["unet"] = ["UNet2DConditionLoadersMixin"]
    _import_structure["utils"] = ["AttnProcsLayers"]
    if is_transformers_available():
        _import_structure["single_file"] = ["FromSingleFileMixin"]
        _import_structure["lora"] = ["LoraLoaderMixin", "StableDiffusionXLLoraLoaderMixin"]
        _import_structure["textual_inversion"] = ["TextualInversionLoaderMixin"]
        _import_structure["ip_adapter"] = ["IPAdapterMixin"]

_import_structure["peft"] = ["PeftAdapterMixin"]


if TYPE_CHECKING or DIFFUSERS_SLOW_IMPORT:
    if is_torch_available():
        from .single_file_model import FromOriginalModelMixin
        from .unet import UNet2DConditionLoadersMixin
        from .utils import AttnProcsLayers

        if is_transformers_available():
            from .ip_adapter import IPAdapterMixin
            from .lora import LoraLoaderMixin, StableDiffusionXLLoraLoaderMixin
            from .single_file import FromSingleFileMixin
            from .textual_inversion import TextualInversionLoaderMixin

    from .peft import PeftAdapterMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)