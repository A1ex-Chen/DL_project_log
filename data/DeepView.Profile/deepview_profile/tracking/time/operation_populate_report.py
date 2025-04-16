def populate_report(self, builder):
    for op_info in self.operations:
        builder.add_run_time_entry(operation_name=remove_dunder(op_info.
            operation_name), forward_ms=op_info.forward_ms, backward_ms=
            op_info.backward_ms, stack_context=op_info.stack)
