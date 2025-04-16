def _get_all_paths_util(graph, u, d, visited, path):
    visited[u] = True
    path.append(u)
    if u == d:
        all_paths.append(copy.deepcopy(path))
    else:
        for i in graph[u]:
            if not visited[i]:
                _get_all_paths_util(graph, i, d, visited, path)
    path.pop()
    visited[u] = False
