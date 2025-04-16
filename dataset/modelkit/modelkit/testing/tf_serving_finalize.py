def finalize():
    subprocess.call(['docker', 'kill', 'modelkit-tfserving-tests'])
    tfserving_proc.terminate()
