def unwrap_schedule(scheduler, num_steps=10):
    lrs = []
    for _ in range(num_steps):
        scheduler.step()
        lrs.append(scheduler.get_lr())
    return lrs
