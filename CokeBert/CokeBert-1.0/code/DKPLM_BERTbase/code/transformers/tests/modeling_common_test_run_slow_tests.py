@pytest.mark.slow
def run_slow_tests(self):
    self.create_and_check_model_from_pretrained()
