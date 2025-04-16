def print_memory_trace_statistics(self, summary: MemorySummary):
    self.print_fn('\nLine by line memory consumption:\n' + '\n'.join(
        f'{state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}'
         for state in summary.sequential))
    self.print_fn('\nLines with top memory consumption:\n' + '\n'.join(
        f'=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}'
         for state in summary.cumulative[:6]))
    self.print_fn('\nLines with lowest memory consumption:\n' + '\n'.join(
        f'=> {state.frame.filename}:{state.frame.line_number}: mem {state.cpu_gpu}: {state.frame.line_text}'
         for state in summary.cumulative[-6:]))
    self.print_fn(f'\nTotal memory increase: {summary.total}')
