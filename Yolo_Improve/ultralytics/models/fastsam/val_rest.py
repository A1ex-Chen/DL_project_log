# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import SegmentMetrics


class FastSAMValidator(SegmentationValidator):
    """
    Custom validation class for fast SAM (Segment Anything Model) segmentation in Ultralytics YOLO framework.

    Extends the SegmentationValidator class, customizing the validation process specifically for fast SAM. This class
    sets the task to 'segment' and uses the SegmentMetrics for evaluation. Additionally, plotting features are disabled
    to avoid errors during validation.

    Attributes:
        dataloader: The data loader object used for validation.
        save_dir (str): The directory where validation results will be saved.
        pbar: A progress bar object.
        args: Additional arguments for customization.
        _callbacks: List of callback functions to be invoked during validation.
    """
