def format_help(self):
    help_msg = str(self.description)
    return help_msg + ', available arguments: ' + self.format_arguments()
