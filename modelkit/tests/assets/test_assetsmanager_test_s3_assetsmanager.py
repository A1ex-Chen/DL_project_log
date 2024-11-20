@skip_unless('ENABLE_S3_TEST', 'True')
def test_s3_assetsmanager(s3_assetsmanager):
    _perform_mng_test(s3_assetsmanager)
