def _load_json_manifest(self, fpath):
    j = json.load(open(fpath, 'r', encoding='utf-8'))
    for i, s in enumerate(j):
        if i % 1000 == 0:
            print(f'{i:>10}/{len(j):<10}', end='\r')
        s_max_duration = s['original_duration']
        s['duration'] = s.pop('original_duration')
        if not self.min_duration <= s_max_duration <= self.max_duration:
            self.duration_filtered += s['duration']
            continue
        tr = s.get('transcript', None) or self.load_transcript(s[
            'text_filepath'])
        if not isinstance(tr, str):
            print(f'WARNING: Skipped sample (transcript not a str): {tr}.')
            self.duration_filtered += s['duration']
            continue
        if self.normalize_transcripts:
            tr = normalize_string(tr, self.tokenizer.charset, self.
                punctuation_map)
        s['transcript'] = self.tokenizer.tokenize(tr)
        files = s.pop('files')
        if self.ignore_offline_speed_perturbation:
            files = [f for f in files if f['speed'] == 1.0]
        s['audio_duration'] = [f['duration'] for f in files]
        s['audio_filepath'] = [str(Path(self.data_dir, f['fname'])) for f in
            files]
        self.samples.append(s)
        self.duration += s['duration']
        if self.max_utts > 0 and len(self.samples) >= self.max_utts:
            print(
                f'Reached max_utts={self.max_utts}. Finished parsing {fpath}.')
            break
