def notify_slack(webhook_url, library_name, version, release_info):
    """Send a notification to a Slack channel."""
    message = f"""🚀 New release for {library_name} available: version **{version}** 🎉
📜 Release Notes: {release_info['url']}
⏱️ Release time: {release_info['release_time']}"""
    payload = {'text': message}
    response = requests.post(webhook_url, json=payload)
    if response.status_code == 200:
        print('Notification sent to Slack successfully.')
    else:
        print('Failed to send notification to Slack.')
