def print_and_log(*args):
    with open(self.args.log_filename, 'a') as log_file:
        log_file.write(''.join(args) + '\n')
    print(*args)
