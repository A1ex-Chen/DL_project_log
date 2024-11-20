from dataclasses import field


try:
    from torchreid.utils import FeatureExtractor
except ImportError:
    print(
        "The torchreid module is not installed. Please install it using the following command:\n"
        "pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
    )

from sportslabkit.constants import CACHE_DIR
from sportslabkit.image_model.base import BaseImageModel
from sportslabkit.logger import logger
from sportslabkit.utils import (
    HiddenPrints,
    download_file_from_google_drive,
)


model_save_dir = CACHE_DIR / "sportslabkit" / "models" / "torchreid"

model_dict = {
    "shufflenet": "https://drive.google.com/file/d/1RFnYcHK1TM-yt3yLsNecaKCoFO4Yb6a-/view?usp=sharing",
    "mobilenetv2_x1_0": "https://drive.google.com/file/d/1K7_CZE_L_Tf-BRY6_vVm0G-0ZKjVWh3R/view?usp=sharing",
    "mobilenetv2_x1_4": "https://drive.google.com/file/d/10c0ToIGIVI0QZTx284nJe8QfSJl5bIta/view?usp=sharing",
    "mlfn": "https://drive.google.com/file/d/1PP8Eygct5OF4YItYRfA3qypYY9xiqHuV/view?usp=sharing",
    "osnet_x1_0": "https://drive.google.com/file/d/1LaG1EJpHrxdAxKnSCJ_i0u-nbxSAeiFY/view?usp=sharing",
    "osnet_x0_75": "https://drive.google.com/file/d/1uwA9fElHOk3ZogwbeY5GkLI6QPTX70Hq/view?usp=sharing",
    "osnet_x0_5": "https://drive.google.com/file/d/16DGLbZukvVYgINws8u8deSaOqjybZ83i/view?usp=sharing",
    "osnet_x0_25": "https://drive.google.com/file/d/1rb8UN5ZzPKRc_xvtHlyDh-cSz88YX9hs/view?usp=sharing",
    "osnet_ibn_x1_0": "https://drive.google.com/file/d/1sr90V6irlYYDd4_4ISU2iruoRG8J__6l/view?usp=sharing",
    "osnet_ain_x1_0": "https://drive.google.com/file/d/1-CaioD9NaqbHK_kzSMW8VE4_3KcsRjEo/view?usp=sharing",
    "osnet_ain_x0_75": "https://drive.google.com/file/d/1apy0hpsMypqstfencdH-jKIUEFOW4xoM/view?usp=sharing",
    "osnet_ain_x0_5": "https://drive.google.com/file/d/1KusKvEYyKGDTUBVRxRiz55G31wkihB6l/view?usp=sharing",
    "osnet_ain_x0_25": "https://drive.google.com/file/d/1SxQt2AvmEcgWNhaRb2xC4rP6ZwVDP0Wt/view?usp=sharing",
    "resnet50_MSMT17": "https://drive.google.com/file/d/1yiBteqgIZoOeywE8AhGmEQl7FTVwrQmf/view?usp=sharing",
    "osnet_x1_0_MSMT17": "https://drive.google.com/file/d/1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x/view?usp=sharing",
    "osnet_ain_x1_0_MSMT17": "https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view?usp=sharing",
    "resnet50_MSMT17x": "https://drive.google.com/file/d/1ep7RypVDOthCRIAqDnn4_N-UhkkFHJsj/view?usp=sharing",
    "resnet50_fc512_MSMT17": "https://drive.google.com/file/d/1fDJLcz4O5wxNSUvImIIjoaIF9u1Rwaud/view?usp=sharing",
}






class BaseTorchReIDModel(BaseImageModel):




class ShuffleNet(BaseTorchReIDModel):


class MobileNetV2_x1_0(BaseTorchReIDModel):


class MobileNetV2_x1_4(BaseTorchReIDModel):


class MLFN(BaseTorchReIDModel):


class OSNet_x1_0(BaseTorchReIDModel):


class OSNet_x0_75(BaseTorchReIDModel):


class OSNet_x0_5(BaseTorchReIDModel):


class OSNet_x0_25(BaseTorchReIDModel):


class OSNet_ibn_x1_0(BaseTorchReIDModel):


class OSNet_ain_x1_0(BaseTorchReIDModel):


class OSNet_ain_x0_75(BaseTorchReIDModel):


class OSNet_ain_x0_5(BaseTorchReIDModel):


class OSNet_ain_x0_25(BaseTorchReIDModel):


class ResNet50(BaseTorchReIDModel):


class ResNet50_fc512(BaseTorchReIDModel):