import pydantic

from modelkit.core.model import Model


class ItemModel(pydantic.BaseModel):
    x: int


class SomeBadValidatedModel(Model[ItemModel, ItemModel]):


m = SomeBadValidatedModel()
y: int = m(ItemModel(x=10))  # here mypy expects an ItemModel to be returned