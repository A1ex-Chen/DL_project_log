def unwrap_and_save_reload_schedule(scheduler, num_steps=10):
    lrs = []
    for step in range(num_steps):
        scheduler.step()
        lrs.append(scheduler.get_lr())
        if step == num_steps // 2:
            with TemporaryDirectory() as tmpdirname:
                file_name = os.path.join(tmpdirname, 'schedule.bin')
                torch.save(scheduler.state_dict(), file_name)
                state_dict = torch.load(file_name)
                scheduler.load_state_dict(state_dict)
    return lrs
