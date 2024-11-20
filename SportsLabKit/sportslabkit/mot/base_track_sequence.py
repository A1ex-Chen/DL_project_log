@with_callbacks
def track_sequence(self, sequence):
    with tqdm(range(0, len(sequence) - self.window_size + 1, self.step_size
        ), desc='Tracking Progress') as t:
        for i in t:
            self.process_sequence_item(sequence[i:i + self.window_size].
                squeeze())
            t.set_postfix_str(
                f'Active: {len(self.alive_tracklets)}, Dead: {len(self.dead_tracklets)}'
                , refresh=True)
