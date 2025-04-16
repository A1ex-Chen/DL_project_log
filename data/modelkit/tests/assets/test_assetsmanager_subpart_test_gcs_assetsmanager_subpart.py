@skip_unless('ENABLE_GCS_TEST', 'True')
def test_gcs_assetsmanager_subpart(gcs_assetsmanager):
    _perform_mng_test_subpart(gcs_assetsmanager)
