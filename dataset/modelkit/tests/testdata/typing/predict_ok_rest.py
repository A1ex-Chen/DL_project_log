from typing import List

from modelkit.core.model import Model


class OKModel(Model[int, int]):


m = OKModel()
y: int = m(1)
y2: List[int] = m.predict_batch([1])