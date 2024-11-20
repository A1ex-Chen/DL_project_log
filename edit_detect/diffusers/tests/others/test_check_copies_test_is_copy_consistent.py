def test_is_copy_consistent(self):
    self.check_copy_consistency(
        '# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput'
        , 'DDPMSchedulerOutput', REFERENCE_CODE + '\n')
    self.check_copy_consistency(
        '# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput'
        , 'DDPMSchedulerOutput', REFERENCE_CODE)
    self.check_copy_consistency(
        '# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->Test'
        , 'TestSchedulerOutput', re.sub('DDPM', 'Test', REFERENCE_CODE))
    long_class_name = (
        'TestClassWithAReallyLongNameBecauseSomePeopleLikeThatForSomeReason')
    self.check_copy_consistency(
        f'# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->{long_class_name}'
        , f'{long_class_name}SchedulerOutput', re.sub('Bert',
        long_class_name, REFERENCE_CODE))
    self.check_copy_consistency(
        '# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->Test'
        , 'TestSchedulerOutput', REFERENCE_CODE, overwrite_result=re.sub(
        'DDPM', 'Test', REFERENCE_CODE))
