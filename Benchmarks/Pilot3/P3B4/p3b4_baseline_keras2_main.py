def main():
    gParameters = initialize_parameters()
    avg_loss = run(gParameters)
    print('Return: ', avg_loss.history['val_loss'][-1])
