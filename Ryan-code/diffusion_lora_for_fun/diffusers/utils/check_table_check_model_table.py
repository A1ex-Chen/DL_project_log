def check_model_table(overwrite=False):
    """Check the model table in the index.rst is consistent with the state of the lib and maybe `overwrite`."""
    current_table, start_index, end_index, lines = _find_text_in_file(filename
        =os.path.join(PATH_TO_DOCS, 'index.md'), start_prompt=
        '<!--This table is updated automatically from the auto modules',
        end_prompt='<!-- End table-->')
    new_table = get_model_table_from_auto_modules()
    if current_table != new_table:
        if overwrite:
            with open(os.path.join(PATH_TO_DOCS, 'index.md'), 'w', encoding
                ='utf-8', newline='\n') as f:
                f.writelines(lines[:start_index] + [new_table] + lines[
                    end_index:])
        else:
            raise ValueError(
                'The model table in the `index.md` has not been updated. Run `make fix-copies` to fix this.'
                )
