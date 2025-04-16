from typing import Any, Dict

from modelkit.core.library import ModelLibrary
from modelkit.core.model import Model


class SomeModel(Model[str, str]):
    CONFIGURATIONS: Dict[str, Any] = {"dependent": {}}



class SomeOtherModel(Model[int, int]):
    CONFIGURATIONS: Dict[str, Any] = {
        "something": {"model_dependencies": {"dependent"}}
    }



lib = ModelLibrary(models=[SomeModel, SomeOtherModel])

m2 = lib.get("something", model_type=SomeOtherModel)
m2.predict("str")