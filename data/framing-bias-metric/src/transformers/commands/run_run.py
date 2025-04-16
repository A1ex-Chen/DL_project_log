def run(self):
    nlp, outputs = self._nlp, []
    for entry in self._reader:
        output = nlp(**entry) if self._reader.is_multi_columns else nlp(entry)
        if isinstance(output, dict):
            outputs.append(output)
        else:
            outputs += output
    if self._nlp.binary_output:
        binary_path = self._reader.save_binary(outputs)
        logger.warning(
            'Current pipeline requires output to be in binary format, saving at {}'
            .format(binary_path))
    else:
        self._reader.save(outputs)
