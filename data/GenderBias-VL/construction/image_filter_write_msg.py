def write_msg(self, file: TextIOWrapper, msg):
    file.write(msg)
    file.flush()
