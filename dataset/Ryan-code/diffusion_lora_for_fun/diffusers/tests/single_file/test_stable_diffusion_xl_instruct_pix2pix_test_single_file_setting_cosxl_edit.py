def test_single_file_setting_cosxl_edit(self):
    pipe = self.pipeline_class.from_single_file(self.ckpt_path, config=self
        .repo_id, is_cosxl_edit=True)
    assert pipe.is_cosxl_edit is True
