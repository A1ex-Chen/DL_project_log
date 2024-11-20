def update_performance_data(results: List, batch_size: int,
    performance_partial_file: str):
    row: Dict = {'batch_size': batch_size}
    with open(performance_partial_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for r in reader:
            avg_latency = calculate_average_latency(r)
            row = {**row, **r, 'avg latency': avg_latency}
    results.append(row)
