def _save_pretrained(self, save_directory: Union[str, Path], file_name:
    Optional[str]=None, **kwargs):
    """
        Save a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.onnxruntime.modeling_ort.ORTModel.from_pretrained`] class method. It will always save the
        latest_model_name.

        Arguments:
            save_directory (`str` or `Path`):
                Directory where to save the model file.
            file_name(`str`, *optional*):
                Overwrites the default model file name from `"model.onnx"` to `file_name`. This allows you to save the
                model with a different name.
        """
    model_file_name = file_name if file_name is not None else ONNX_WEIGHTS_NAME
    src_path = self.model_save_dir.joinpath(self.latest_model_name)
    dst_path = Path(save_directory).joinpath(model_file_name)
    try:
        shutil.copyfile(src_path, dst_path)
    except shutil.SameFileError:
        pass
    src_path = self.model_save_dir.joinpath(ONNX_EXTERNAL_WEIGHTS_NAME)
    if src_path.exists():
        dst_path = Path(save_directory).joinpath(ONNX_EXTERNAL_WEIGHTS_NAME)
        try:
            shutil.copyfile(src_path, dst_path)
        except shutil.SameFileError:
            pass
