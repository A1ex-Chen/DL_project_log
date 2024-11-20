def train_model(self, batch, current_step=0, pbar=None):
    self.ds_engine.module.train()
    loss, mle_acc = self.ds_engine(batch)
    self.ds_engine.backward(loss)
    self.ds_engine.step()
    pbar.set_description(
        f'[!] loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}'
        )
    pbar.update(1)
    if self.args['local_rank'] == 0 and self.args['log_path'
        ] and current_step % self.args['logging_step'] == 0:
        elapsed = pbar.format_dict['elapsed']
        rate = pbar.format_dict['rate']
        remaining = (pbar.total - pbar.n) / rate if rate and pbar.total else 0
        remaining = str(datetime.timedelta(seconds=remaining))
        self.writer.add_scalar('train/loss', loss.item(), current_step)
        self.writer.add_scalar('train/token_acc', mle_acc * 100, current_step)
        logging.info(
            f'[!] progress: {round(pbar.n / pbar.total, 5)}; remaining time: {remaining}; loss: {round(loss.item(), 4)}; token_acc: {round(mle_acc * 100, 2)}'
            )
    mle_acc *= 100
    return mle_acc
