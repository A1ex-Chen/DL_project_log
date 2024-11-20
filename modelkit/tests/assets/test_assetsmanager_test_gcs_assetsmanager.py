@skip_unless('ENABLE_GCS_TEST', 'True')
def test_gcs_assetsmanager(gcs_assetsmanager):
    _perform_mng_test(gcs_assetsmanager)
