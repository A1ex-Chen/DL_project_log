def random_dropping(pc, e):
    up_num = max(64, 768 // (e // 50 + 1))
    pc = pc
    random_num = torch.randint(1, up_num, (1, 1))[0, 0]
    pc = fps(pc, random_num)
    padding = torch.zeros(pc.size(0), 2048 - pc.size(1), 3).to(pc.device)
    pc = torch.cat([pc, padding], dim=1)
    return pc
