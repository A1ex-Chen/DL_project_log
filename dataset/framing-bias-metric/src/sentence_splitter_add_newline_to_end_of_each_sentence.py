def add_newline_to_end_of_each_sentence(x: str) ->str:
    """This was added to get rougeLsum scores matching published rougeL scores for BART and PEGASUS."""
    re.sub('<n>', '', x)
    assert NLTK_AVAILABLE, 'nltk must be installed to separate newlines between sentences. (pip install nltk)'
    return '\n'.join(nltk.sent_tokenize(x))
