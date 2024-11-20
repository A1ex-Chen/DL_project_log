def evaluate(self, parameters, config):
    print(
        f"Global epoch {config['server_round']}: [Client {self.cid}] evaluate, config: {config}"
        )
    self.model.to(self.device)
    print(
        f'Client {self.cid}: test dataset size {len(self.tesloader)}, Starting validation ...'
        )
    loss, fid = test(self.model, self.tesloader, device=self.device)
    print(
        f'Client {self.cid}: Test FID: {fid}, Test loss: {loss}. Validation end ...'
        )
    return float(loss), len(self.tesloader), {'fid': float(fid)}
