class StorageDriverError(Exception):
    pass


class ObjectDoesNotExistError(StorageDriverError):


class AssetsManagerError(Exception):
    pass


class AssetAlreadyExistsError(AssetsManagerError):


class AssetDoesNotExistError(AssetsManagerError):


class AssetMajorVersionDoesNotExistError(AssetsManagerError):


class InvalidAssetSpecError(AssetsManagerError):


class InvalidVersionError(InvalidAssetSpecError):


class InvalidNameError(InvalidAssetSpecError):


class LocalAssetDoesNotExistError(AssetsManagerError):


class UnknownAssetsVersioningSystemError(AssetsManagerError):
    pass