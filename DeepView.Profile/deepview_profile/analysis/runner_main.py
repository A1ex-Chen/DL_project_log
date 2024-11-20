def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('entry_point', type=str)
    args = parser.parse_args()
    project_root = os.getcwd()
    with NVML() as nvml:
        analyzer = analyze_project(project_root, args.entry_point, nvml)
        breakdown = next(analyzer)
        throughput = next(analyzer)
    print('Peak usage:   ', breakdown.peak_usage_bytes, 'bytes')
    print('Max. capacity:', breakdown.memory_capacity_bytes, 'bytes')
    print('No. of weight breakdown nodes:   ', len(breakdown.operation_tree))
    print('No. of operation breakdown nodes:', len(breakdown.weight_tree))
    print('Throughput:', throughput.samples_per_second, 'samples/s')
