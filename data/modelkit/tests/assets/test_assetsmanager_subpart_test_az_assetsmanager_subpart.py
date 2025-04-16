@skip_unless('ENABLE_AZ_TEST', 'True')
def test_az_assetsmanager_subpart(az_assetsmanager):
    _perform_mng_test_subpart(az_assetsmanager)
