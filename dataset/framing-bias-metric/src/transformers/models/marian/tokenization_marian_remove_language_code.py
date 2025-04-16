def remove_language_code(self, text: str):
    """Remove language codes like <<fr>> before sentencepiece"""
    match = self.language_code_re.match(text)
    code: list = [match.group(0)] if match else []
    return code, self.language_code_re.sub('', text)
