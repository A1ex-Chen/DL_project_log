def evaluate(eval_iter, model, meters, log_interval, max_size=None, repeat=1):
    total_len, total_loss = 0, 0.0
    eval_step = 0
    log_throughput = 0
    log_latency = 0
    log_loss = 0
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        mems = None
        for _ in range(repeat):
            for idx, (data, target, seq_len, warm) in enumerate(eval_iter):
                if max_size and idx >= max_size:
                    break
                eval_step += 1
                torch.cuda.synchronize()
                start_iter = time.time()
                loss, mems = model(data, target, mems)
                torch.cuda.synchronize()
                elapsed = time.time() - start_iter
                loss = loss.float().mean()
                log_loss += loss.item()
                if warm:
                    total_loss += seq_len * loss.item()
                    total_len += seq_len
                meters['eval_latency'].update(elapsed)
                log_latency += elapsed
                target_tokens = target.numel()
                throughput = target_tokens / elapsed
                throughput = utils.distributed.all_reduce_item(throughput,
                    op='sum')
                meters['eval_throughput'].update(throughput)
                log_throughput += throughput
                if eval_step % log_interval == 0:
                    log_throughput /= log_interval
                    log_latency /= log_interval
                    log_loss /= log_interval
                    log_ppl = math.exp(log_loss)
                    log_str = (
                        '| step {:>8d} | batches {:>6d} / {:d} | ms/batch {:5.2f} | tok/s {:7.0f} | loss {:5.2f} | ppl {:5.2f}'
                        .format(eval_step, idx + 1, eval_iter.n_batch, 
                        log_latency * 1000, log_throughput, log_loss, log_ppl))
                    logging.info(log_str)
                    dllogger_data = {'eval_latency': log_latency * 1000,
                        'eval_throughput': log_throughput, 'eval_loss':
                        log_loss, 'eval_perplexity': log_ppl}
                    dllogger.log(step=tuple([eval_step]), data=dllogger_data)
                    log_throughput = 0
                    log_latency = 0
                    log_loss = 0
    utils.distributed.barrier()
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    logging.info('Time : {:.2f}s, {:.2f}ms/segment'.format(total_time, 1000 *
        total_time / (idx + 1)))
    avg_loss = total_loss / total_len
    avg_loss = utils.distributed.all_reduce_item(avg_loss, op='mean')
    return avg_loss
