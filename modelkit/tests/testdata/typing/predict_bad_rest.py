from modelkit.core.model import Model


class BadModel(Model[int, int]):


m = BadModel()
y = m("some string")