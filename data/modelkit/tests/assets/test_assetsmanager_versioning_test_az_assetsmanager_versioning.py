@skip_unless('ENABLE_AZ_TEST', 'True')
def test_az_assetsmanager_versioning(az_assetsmanager):
    _perform_mng_test(az_assetsmanager)
