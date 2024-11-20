def last_checkpoint(self):
    tracked = list(self.tracked.values())
    if len(tracked) >= 1:
        try:
            torch.load(tracked[-1], map_location='cpu')
            return tracked[-1]
        except:
            print_once(f'Last checkpoint {tracked[-1]} appears corrupted.')
    elif len(tracked) >= 2:
        return tracked[-2]
    else:
        return None
