def check_imshow(warn=False):
    try:
        assert not is_jupyter()
        assert not is_docker()
        cv2.imshow('test', np.zeros((1, 1, 3)))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(
                f"""WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()
{e}"""
                )
        return False
