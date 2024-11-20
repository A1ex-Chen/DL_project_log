def _new_step(self):
    """ Internal method that resets the variable for a new step. """
    self.should_save = False
    self.should_evaluate = False
    self.should_log = False
