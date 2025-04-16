def global_version_update(version, patch=False):
    """Update the version in all needed files."""
    for pattern, fname in REPLACE_FILES.items():
        update_version_in_file(fname, version, pattern)
    if not patch:
        update_version_in_examples(version)
