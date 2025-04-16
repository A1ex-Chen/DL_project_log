def write_to_csv(file_name: str, data_dict: Dict[str, Union[str, bool, float]]
    ):
    """Serializes a dictionary into a CSV file."""
    with open(file_name, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        writer.writerow(data_dict)
