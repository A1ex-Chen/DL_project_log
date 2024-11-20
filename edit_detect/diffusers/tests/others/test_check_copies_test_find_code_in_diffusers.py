def test_find_code_in_diffusers(self):
    code = check_copies.find_code_in_diffusers(
        'schedulers.scheduling_ddpm.DDPMSchedulerOutput')
    self.assertEqual(code, REFERENCE_CODE)
