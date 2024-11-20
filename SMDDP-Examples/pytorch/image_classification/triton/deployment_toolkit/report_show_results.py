def show_results(results: List[Dict]):
    headers = list(results[0].keys())
    summary = map(lambda x: list(map(lambda item: item[1], x.items())), results
        )
    print(tabulate(summary, headers=headers))
