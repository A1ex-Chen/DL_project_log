def collate_csv(input_files: List[str], output_file: str):
    """Collates multiple identically structured CSVs into a single CSV file."""
    with open(output_file, mode='w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=BENCHMARK_FIELDS)
        writer.writeheader()
        for file in input_files:
            with open(file, mode='r') as infile:
                reader = csv.DictReader(infile)
                for row in reader:
                    writer.writerow(row)
