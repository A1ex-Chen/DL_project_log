def write_to_csv(image_name, prediction, confidence):
    data = {'Image Name': image_name, 'Prediction': prediction,
        'Confidence': confidence}
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not csv_path.is_file():
            writer.writeheader()
        writer.writerow(data)
