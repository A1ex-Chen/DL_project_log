def check_imshow(warn=False):
    """Check if environment supports image displays."""
    try:
        if LINUX:
            assert not IS_COLAB and not IS_KAGGLE
            assert 'DISPLAY' in os.environ, "The DISPLAY environment variable isn't set."
        cv2.imshow('create_self_data', np.zeros((8, 8, 3), dtype=np.uint8))
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
