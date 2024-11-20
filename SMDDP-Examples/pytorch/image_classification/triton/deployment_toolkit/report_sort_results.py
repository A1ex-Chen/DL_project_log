def sort_results(results: List):
    results = natsorted(results, key=lambda item: [item[key] for key in
        item.keys()])
    return results
