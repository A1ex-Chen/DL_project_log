@app.route(DETECTION_URL, methods=['POST'])
def predict():
    if request.method != 'POST':
        return
    if request.files.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        results = model(im, size=640)
        return results.pandas().xyxy[0].to_json(orient='records')
