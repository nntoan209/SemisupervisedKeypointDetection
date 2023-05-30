from torch.utils.data import DataLoader

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, weight):
        self._add(val, weight)

    def _add(self, val, weight):
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def average(self):
        return self.avg

    def accumulate(self):
        return self.sum
    
class InfiniteDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(InfiniteDataLoader, self).__init__(*args, **kwargs)
        self.iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = super().__iter__()
            batch = next(self.iterator)
        return batch
    
def ema_decay_scheduler(start_ema_decay, end_ema_decay, max_step, step):
    if step > max_step:
        return end_ema_decay
    else:
        return start_ema_decay + (end_ema_decay - start_ema_decay) / max_step * step
    