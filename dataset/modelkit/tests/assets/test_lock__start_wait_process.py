def _start_wait_process(lock_path, duration_s):
    script_path = os.path.join(TEST_DIR, 'assets', 'resources', 'lock.py')
    result = None

    def run():
        nonlocal result
        try:
            cmd = [sys.executable, script_path, lock_path, str(duration_s)]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=
                subprocess.STDOUT)
            stdout, _ = p.communicate()
            stdout = stdout.decode('utf-8')
            if p.returncode:
                print('ERROR', p.returncode, stdout, flush=True)
                raise Exception('lock.py failed')
            result = stdout
        except Exception:
            traceback.print_exc()
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()

    def join():
        t.join()
        return result
    return join
