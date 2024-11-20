def setup_source(self, source):
    """
        Sets up the data source for inference.

        This method configures the data source from which images will be fetched for inference. The source could be a
        directory, a video file, or other types of image data sources.

        Args:
            source (str | Path): The path to the image data source for inference.
        """
    if source is not None:
        super().setup_source(source)
