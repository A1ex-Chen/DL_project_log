def evaluate(self, predictions: List[Prediction], request_metrics=None):
    _, predictions_filename = tempfile.mkstemp(suffix='.json', text=True)
    json.dump(predictions, open(predictions_filename, 'w'))
    submission_command = (
        f'evalai challenge {self._challenge_id} phase {self._phase_id} submit --file {predictions_filename}'
        )
    submission_command_subprocess = subprocess.Popen(submission_command.
        split(), stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=
        subprocess.STDOUT)
    submission_command_stdout = submission_command_subprocess.communicate(input
        =b'N\n')[0].decode('utf-8')
    submission_id_regex = re.search('evalai submission ([0-9]+)',
        submission_command_stdout)
    submission_id = submission_id_regex.group(0).split()[-1]
    result_stdout: str = 'The Submission is yet to be evaluated.'
    num_tries: int = 0
    while 'CIDEr' not in result_stdout:
        time.sleep(10)
        result_stdout = subprocess.check_output(['evalai', 'submission',
            submission_id, 'result']).decode('utf-8')
        num_tries += 1
        if num_tries == 60:
            raise ConnectionError(
                'Unable to get results from EvalAI within 10 minutes!')
    metrics = json.loads(result_stdout, encoding='utf-8')
    metrics = {'in-domain': metrics[0]['in-domain'], 'near-domain': metrics
        [1]['near-domain'], 'out-domain': metrics[2]['out-domain'],
        'entire': metrics[3]['entire']}
    flipped_metrics: Dict[str, Any] = defaultdict(dict)
    for key, val in metrics.items():
        for subkey, subval in val.items():
            flipped_metrics[subkey][key] = subval
    metrics = flipped_metrics
    return metrics
