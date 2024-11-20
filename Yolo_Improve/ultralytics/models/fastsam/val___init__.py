def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None,
    _callbacks=None):
    """
        Initialize the FastSAMValidator class, setting the task to 'segment' and metrics to SegmentMetrics.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.

        Notes:
            Plots for ConfusionMatrix and other related metrics are disabled in this class to avoid errors.
        """
    super().__init__(dataloader, save_dir, pbar, args, _callbacks)
    self.args.task = 'segment'
    self.args.plots = False
    self.metrics = SegmentMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
