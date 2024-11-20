def populate_report(self, builder):
    for entry in self._activations:
        builder.add_activation_entry(operation_name=remove_dunder(entry.
            operation_name), size_bytes=entry.size_bytes, stack_context=
            entry.stack)
