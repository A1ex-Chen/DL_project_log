@app.route(DETECTION_URL, methods=['POST'])
def predict(model):
    if request.method != 'POST':
        return
    if request.files.get('image'):
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        if model in models:
            results = models[model](im, size=640)
            return results.pandas().xyxy[0].to_json(orient='records')
