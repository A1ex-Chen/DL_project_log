from modelkit.core.model import Asset


class BaseAsset(Asset):


class DerivedAsset(BaseAsset):
    CONFIGURATIONS = {"derived_asset": {"asset": "something.txt"}}