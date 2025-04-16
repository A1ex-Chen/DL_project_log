@click.command()
@click.argument('lock_path')
@click.argument('duration_s', type=float)
def wait(lock_path, duration_s):
    """Take a lock, wait a bit, release the lock

    And print the acquisition and release time, as well as wait loops.
    """
    with filelock.FileLock(lock_path, 3 * 60):
        print(time.time())
        time.sleep(duration_s)
        print(time.time())
