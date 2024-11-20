@skip_unless('ENABLE_TF_TEST', 'True')
def test_configure_package():
    sys.path.append(os.path.join(TEST_DIR, 'testdata'))
    confs = configure(models='test_module')
    assert len(confs) == 5
