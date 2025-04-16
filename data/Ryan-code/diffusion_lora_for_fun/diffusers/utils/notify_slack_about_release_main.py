def main():
    latest_version = check_pypi_for_latest_release(LIBRARY_NAME)
    release_info = get_github_release_info(GITHUB_REPO)
    parsed_version = release_info['tag_name'].replace('v', '')
    if latest_version and release_info and latest_version == parsed_version:
        notify_slack(SLACK_WEBHOOK_URL, LIBRARY_NAME, latest_version,
            release_info)
    else:
        raise ValueError('There were some problems.')
