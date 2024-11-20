def update_performance_data(results: List, performance_file: str):
    with open(performance_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row['avg latency'] = calculate_average_latency(row)
            results.append(row)
