def read_img_ceph(img_path):
    init_ceph_client_if_needed()
    img_bytes = client.get(img_path)
    assert img_bytes is not None, f'Please check image at {img_path}'
    img_mem_view = memoryview(img_bytes)
    img_array = np.frombuffer(img_mem_view, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return img
