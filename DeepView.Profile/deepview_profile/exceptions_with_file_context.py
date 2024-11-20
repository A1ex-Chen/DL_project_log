def with_file_context(self, file_path, line_number=None):
    self.file_context = FileContext(file_path=file_path, line_number=
        line_number)
    return self
