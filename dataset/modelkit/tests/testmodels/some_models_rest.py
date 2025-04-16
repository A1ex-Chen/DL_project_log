from modelkit.core.model import Model


class BaseModel(Model):


class DerivedModel(BaseModel):
    CONFIGURATIONS = {"derived_model": {"asset": "something.txt"}}
