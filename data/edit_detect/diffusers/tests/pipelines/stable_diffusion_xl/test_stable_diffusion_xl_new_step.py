def new_step(self, *args, **kwargs):
    done_steps.append(args[1].cpu().item())
    return old_step(self, *args, **kwargs)
