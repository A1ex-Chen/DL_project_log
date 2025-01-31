def get_output_file_path(self) ->str:
    """
        Calculate the output path being used in case you are writing your frames to a video file.

        Useful if you didn't set `output_path`, and want to know what the autogenerated output file path by Norfair will be.

        Returns
        -------
        str
            The path to the file.
        """
    if not os.path.isdir(self.output_path):
        return self.output_path
    if self.input_path is not None:
        file_name = self.input_path.split('/')[-1].split('.')[0]
    else:
        file_name = 'camera_{self.camera}'
    file_name = f'{file_name}_out.{self.output_extension}'
    return os.path.join(self.output_path, file_name)
