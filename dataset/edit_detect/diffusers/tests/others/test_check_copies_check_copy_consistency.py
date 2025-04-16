def check_copy_consistency(self, comment, class_name, class_code,
    overwrite_result=None):
    code = comment + f'\nclass {class_name}(nn.Module):\n' + class_code
    if overwrite_result is not None:
        expected = (comment + f'\nclass {class_name}(nn.Module):\n' +
            overwrite_result)
    code = check_copies.run_ruff(code)
    fname = os.path.join(self.diffusers_dir, 'new_code.py')
    with open(fname, 'w', newline='\n') as f:
        f.write(code)
    if overwrite_result is None:
        self.assertTrue(len(check_copies.is_copy_consistent(fname)) == 0)
    else:
        check_copies.is_copy_consistent(f.name, overwrite=True)
        with open(fname, 'r') as f:
            self.assertTrue(f.read(), expected)
