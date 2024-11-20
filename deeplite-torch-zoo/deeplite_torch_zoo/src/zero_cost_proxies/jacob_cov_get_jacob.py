def get_jacob(model, model_output_generator, output_post_processing):
    inputs, outputs, _, _ = next(model_output_generator(model,
        input_gradient=True))
    outputs = output_post_processing(outputs)
    outputs.backward(torch.ones_like(outputs))
    jacob = inputs.grad.detach()
    inputs.requires_grad_(False)
    return jacob, outputs.detach()
