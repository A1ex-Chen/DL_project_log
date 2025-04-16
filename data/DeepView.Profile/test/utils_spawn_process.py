def spawn_process(self):
    working_dir = os.path.dirname(self.entry_point)
    entry_filename = os.path.basename(self.entry_point)
    launch_command = ['python', '-m', 'deepview_profile', 'interactive',
        '--debug']
    self.process = subprocess.Popen(launch_command, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, cwd=working_dir)
    self.stdout_thread = threading.Thread(target=stream_monitor, args=(self
        .process.stdout, self.on_message_stdout))
    self.stdout_thread.start()
    self.stderr_thread = threading.Thread(target=stream_monitor, args=(self
        .process.stderr, self.on_message_stderr))
    self.stderr_thread.start()
