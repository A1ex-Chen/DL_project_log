def reorder_incremental_state_scripting(self, incremental_state, new_order):
    for module in incremental_state:
        for key in incremental_state[module]:
            result = incremental_state[module][key].index_select(0, new_order)
            incremental_state[module][key] = result
