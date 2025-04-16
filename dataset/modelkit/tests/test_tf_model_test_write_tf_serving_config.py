@skip_unless('ENABLE_TF_TEST', 'True')
def test_write_tf_serving_config(base_dir, assetsmanager_settings):
    write_config(os.path.join(base_dir, 'test.config'), {'model0':
        '/some/path'})
    ref = testing.ReferenceText(os.path.join(TEST_DIR, 'testdata'))
    with open(os.path.join(base_dir, 'test.config')) as f:
        ref.assert_equal('test.config', f.read())
