@staticmethod
def from_str(format: str, output_path: Optional[str], input_path: Optional[
    str], column: Optional[str], overwrite=False) ->'PipelineDataFormat':
    """
        Creates an instance of the right subclass of :class:`~transformers.pipelines.PipelineDataFormat` depending on
        :obj:`format`.

        Args:
            format: (:obj:`str`):
                The format of the desired pipeline. Acceptable values are :obj:`"json"`, :obj:`"csv"` or :obj:`"pipe"`.
            output_path (:obj:`str`, `optional`):
                Where to save the outgoing data.
            input_path (:obj:`str`, `optional`):
                Where to look for the input data.
            column (:obj:`str`, `optional`):
                The column to read.
            overwrite (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to overwrite the :obj:`output_path`.

        Returns:
            :class:`~transformers.pipelines.PipelineDataFormat`: The proper data format.
        """
    if format == 'json':
        return JsonPipelineDataFormat(output_path, input_path, column,
            overwrite=overwrite)
    elif format == 'csv':
        return CsvPipelineDataFormat(output_path, input_path, column,
            overwrite=overwrite)
    elif format == 'pipe':
        return PipedPipelineDataFormat(output_path, input_path, column,
            overwrite=overwrite)
    else:
        raise KeyError('Unknown reader {} (Available reader are json/csv/pipe)'
            .format(format))
