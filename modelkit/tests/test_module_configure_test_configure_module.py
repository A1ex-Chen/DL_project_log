@skip_unless('ENABLE_TF_TEST', 'True')
def test_configure_module():
    sys.path.append(os.path.join(TEST_DIR, 'testdata'))
    confs = configure(models='test_module.module_a')
    assert len(confs) == 3
