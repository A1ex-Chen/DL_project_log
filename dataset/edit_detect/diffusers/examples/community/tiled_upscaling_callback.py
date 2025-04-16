def callback(obj):
    print(f"progress: {obj['progress']:.4f}")
    obj['image'].save('diffusers_library_progress.jpg')
