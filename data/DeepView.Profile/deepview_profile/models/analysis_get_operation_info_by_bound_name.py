def get_operation_info_by_bound_name(self, bound_name):
    if bound_name not in self.operations:
        return None
    return self.operations[bound_name]
