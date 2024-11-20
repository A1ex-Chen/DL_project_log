def remove_handles(text):
    """
    Remove Twitter username handles from text.
    """
    pattern = regex.compile(
        '(?<![A-Za-z0-9_!@#\\$%&*])@(([A-Za-z0-9_]){20}(?!@))|(?<![A-Za-z0-9_!@#\\$%&*])@(([A-Za-z0-9_]){1,19})(?![A-Za-z0-9_]*@)'
        )
    return pattern.sub(' ', text)
