def main():
    gParameters = initialize_parameters()
    avg_loss = run(gParameters)
    print('Average loss: ', avg_loss)
