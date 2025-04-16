def _save_one_file(file):
    """Save one matplotlib plot to 'file'."""
    plt.savefig(file, dpi=200)
    plt.close()
    LOGGER.info(f'Saved {file}')
