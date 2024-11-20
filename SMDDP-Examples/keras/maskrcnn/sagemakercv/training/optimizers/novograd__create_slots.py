def _create_slots(self, var_list):
    for var in var_list:
        self.add_slot(var=var, slot_name='m', initializer='zeros')
    for var in var_list:
        self.add_slot(var=var, slot_name='v', initializer=tf.zeros(shape=[],
            dtype=var.dtype))
    if self.amsgrad:
        for var in var_list:
            self.add_slot(var, 'vhat')
