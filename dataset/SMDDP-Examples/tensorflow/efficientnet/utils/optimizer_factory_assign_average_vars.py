def assign_average_vars(self, var_list: List[tf.Variable]):
    """Assign variables in var_list with their respective averages.

    Args:
      var_list: List of model variables to be assigned to their average.
    Returns:
      assign_op: The op corresponding to the assignment operation of
        variables to their average.
    """
    assign_op = tf.group([var.assign(self.get_slot(var, 'average')) for var in
        var_list if var.trainable])
    return assign_op
