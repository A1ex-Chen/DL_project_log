def __init__(self, data_dir, manifest_fpaths, tokenizer, sample_rate=16000,
    min_duration=0.1, max_duration=float('inf'), max_utts=0,
    normalize_transcripts=True, trim_silence=False, speed_perturbation=None,
    ignore_offline_speed_perturbation=False):
    """Loads audio, transcript and durations listed in a .json file.

        Args:
            data_dir: absolute path to dataset folder
            manifest_filepath: relative path from dataset folder
                to manifest json as described above. Can be coma-separated paths.
            tokenizer: class converting transcript to tokens
            min_duration (int): skip audio shorter than threshold
            max_duration (int): skip audio longer than threshold
            max_utts (int): limit number of utterances
            normalize_transcripts (bool): normalize transcript text
            trim_silence (bool): trim leading and trailing silence from audio
            ignore_offline_speed_perturbation (bool): use precomputed speed perturbation

        Returns:
            tuple of Tensors
        """
    self.data_dir = data_dir
    self.tokenizer = tokenizer
    self.punctuation_map = punctuation_map(self.tokenizer.charset)
    self.max_utts = max_utts
    self.normalize_transcripts = normalize_transcripts
    self.ignore_offline_speed_perturbation = ignore_offline_speed_perturbation
    self.min_duration = min_duration
    self.max_duration = max_duration
    self.trim_silence = trim_silence
    self.sample_rate = sample_rate
    perturbations = []
    if speed_perturbation is not None:
        perturbations.append(SpeedPerturbation(**speed_perturbation))
    self.perturbations = perturbations
    self.max_duration = max_duration
    self.samples = []
    self.duration = 0.0
    self.duration_filtered = 0.0
    for fpath in manifest_fpaths:
        self._load_json_manifest(fpath)
