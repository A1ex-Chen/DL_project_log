def initialize_parameters():
    mnist_common = mnist.MNIST(mnist.file_path, 'mnist_params.txt', 'keras',
        prog='mnist_cnn', desc='MNIST CNN example')
    gParameters = candle.finalize_parameters(mnist_common)
    return gParameters
