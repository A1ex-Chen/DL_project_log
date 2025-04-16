@classmethod
def create_from_csv(cls, file_name, split_name='', column_label=0,
    column_text=1, column_id=None, skip_first_row=False, **kwargs):
    processor = cls(**kwargs)
    processor.add_examples_from_csv(file_name, split_name=split_name,
        column_label=column_label, column_text=column_text, column_id=
        column_id, skip_first_row=skip_first_row, overwrite_labels=True,
        overwrite_examples=True)
    return processor
