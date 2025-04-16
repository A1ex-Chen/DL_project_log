@staticmethod
def print_table(table_rows):
    """Formats and prints a comparison table for different models with given statistics and performance data."""
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'GPU'
    header = (
        f'| Model | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>{gpu} TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |'
        )
    separator = (
        '|-------------|---------------------|--------------------|------------------------------|-----------------------------------|------------------|-----------------|'
        )
    print(f'\n\n{header}')
    print(separator)
    for row in table_rows:
        print(row)
