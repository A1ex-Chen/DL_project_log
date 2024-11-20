def _create_slots(self, var_list):
    if self._momentum:
        for var in var_list:
            self.add_slot(var, 'momentum')
