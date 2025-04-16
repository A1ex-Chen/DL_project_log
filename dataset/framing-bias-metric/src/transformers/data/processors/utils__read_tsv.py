@classmethod
def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, 'r', encoding='utf-8-sig') as f:
        return list(csv.reader(f, delimiter='\t', quotechar=quotechar))
