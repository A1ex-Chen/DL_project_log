def main():
    branches = fetch_all_branches(USER, REPO)
    filtered_branches = []
    for branch in branches:
        if branch.startswith('v') and ('-release' in branch or '-patch' in
            branch):
            filtered_branches.append(branch)
    sorted_branches = sorted(filtered_branches, key=lambda x: parse(x.split
        ('-')[0][1:]), reverse=True)
    latest_branch = sorted_branches[0]
    return latest_branch
