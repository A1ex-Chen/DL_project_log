def main():
    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo('huggingface/diffusers')
    open_issues = repo.get_issues(state='open')
    for issue in open_issues:
        labels = [label.name.lower() for label in issue.get_labels()]
        if 'stale' in labels:
            comments = sorted(issue.get_comments(), key=lambda i: i.
                created_at, reverse=True)
            last_comment = comments[0] if len(comments) > 0 else None
            if (last_comment is not None and last_comment.user.login !=
                'github-actions[bot]'):
                issue.edit(state='open')
                issue.remove_from_labels('stale')
        elif (dt.now(timezone.utc) - issue.updated_at).days > 23 and (dt.
            now(timezone.utc) - issue.created_at).days >= 30 and not any(
            label in LABELS_TO_EXEMPT for label in labels):
            issue.create_comment(
                """This issue has been automatically marked as stale because it has not had recent activity. If you think this still needs to be addressed please comment on this thread.

Please note that issues that do not follow the [contributing guidelines](https://github.com/huggingface/diffusers/blob/main/CONTRIBUTING.md) are likely to be ignored."""
                )
            issue.add_to_labels('stale')
