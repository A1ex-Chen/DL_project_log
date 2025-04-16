def finalize():
    subprocess.call(['docker', 'rm', '-f', 'redis-tests'])
