def __init__(self, blank_idx, batch_eval_mode, cg_unroll_factor,
    rnnt_config, amp_level, max_symbols_per_step=30, max_symbol_per_sample=None
    ):
    self.blank_idx = blank_idx
    assert max_symbols_per_step is None or max_symbols_per_step > 0
    self.max_symbols = max_symbols_per_step
    assert max_symbol_per_sample is None or max_symbol_per_sample > 0
    self.max_symbol_per_sample = max_symbol_per_sample
    self._SOS = -1
    self.rnnt_config = rnnt_config
    self.cg_captured = False
    self.batch_eval_mode = batch_eval_mode
    self.cg_unroll_factor = cg_unroll_factor
    self.model = None
    self.amp_level = amp_level
