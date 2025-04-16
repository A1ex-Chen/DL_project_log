def predict(interpreter, data):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    input_data = numpy.array(data, dtype=numpy.float32)
    output_data = numpy.empty_like(data)
    for i in range(input_data.shape[0]):
        interpreter.set_tensor(input_details[0]['index'], input_data[i:i + 
            1, :])
        interpreter.invoke()
        output_data[i:i + 1, :] = interpreter.get_tensor(output_details[0][
            'index'])
    return output_data
