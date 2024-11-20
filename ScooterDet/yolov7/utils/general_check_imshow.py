def check_imshow():
    try:
        assert not isdocker(
            ), 'cv2.imshow() is disabled in Docker environments'
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        print(
            f"""WARNING: Environment does not support cv2.imshow() or PIL Image.show() image displays
{e}"""
            )
        return False
