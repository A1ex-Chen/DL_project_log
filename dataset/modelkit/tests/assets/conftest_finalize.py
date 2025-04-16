def finalize():
    subprocess.call(['docker', 'stop', 'modelkit-storage-gcs-tests'])
    minio_proc.terminate()
    minio_proc.wait()
