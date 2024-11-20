def update_version_in_examples(version):
    """Update the version in all examples files."""
    for folder, directories, fnames in os.walk(PATH_TO_EXAMPLES):
        if 'research_projects' in directories:
            directories.remove('research_projects')
        if 'legacy' in directories:
            directories.remove('legacy')
        for fname in fnames:
            if fname.endswith('.py'):
                update_version_in_file(os.path.join(folder, fname), version,
                    pattern='examples')
