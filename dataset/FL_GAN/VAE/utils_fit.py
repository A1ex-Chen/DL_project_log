def fit(self, parameters, config):
    print(
        f"Global epoch {config['server_round']}: [Client {self.cid}] fit, config: {config}"
        )
    self.set_parameters(parameters)
    epochs: int = config['local_epochs']
    print(
        f'Client {self.cid}: train dataset number {len(self.trainloader)}, Starting training ...'
        )
    self.model.to(self.device)
    results = train(self.model, self.trainLoader, epochs, self.
        privacy_engine, self.device)
    print(
        f"Client {self.cid}: Train FID {results['fid']}, Epsilon {results['epsilon']}. Training end ..."
        )
    return self.get_parameters(), len(self.trainloader), results
