def inference(self, im, *args, **kwargs):
    """Runs inference on a given image using the specified model and arguments."""
    visualize = increment_path(self.save_dir / Path(self.batch[0][0]).stem,
        mkdir=True
        ) if self.args.visualize and not self.source_type.tensor else False
    return self.model(im, *args, augment=self.args.augment, visualize=
        visualize, embed=self.args.embed, **kwargs)
