@property
def print_fn(self):
    if self._print_fn is None:
        if self.args.log_print:

            def print_and_log(*args):
                with open(self.args.log_filename, 'a') as log_file:
                    log_file.write(''.join(args) + '\n')
                print(*args)
            self._print_fn = print_and_log
        else:
            self._print_fn = print
    return self._print_fn
