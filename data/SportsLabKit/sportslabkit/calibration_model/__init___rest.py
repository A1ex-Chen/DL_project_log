import inspect

from sportslabkit.calibration_model.base import BaseCalibrationModel
from sportslabkit.calibration_model.dummy import DummyCalibrationModel
from sportslabkit.calibration_model.fld import SimpleContourCalibrator, FLDCalibrator
from sportslabkit.logger import logger


__all__ = ["BaseCalibrationModel", "load", "show_available_models", "DummyCalibrationModel", "LineBasedCalibrator"]








if __name__ == "__main__":
    for cls in inheritors(BaseCalibrationModel):
        print(cls.__name__)