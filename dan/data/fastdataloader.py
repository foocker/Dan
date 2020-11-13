import torch
import threading, queue


class CudaDataLoader:
    """ 
    异步预先将数据从CPU加载到GPU中
    来自： https://www.zhihu.com/question/307282137/answer/1560137140
     """

    def __init__(self, loader, device, queue_size=2):
        self.device = device
        self.queue_size = queue_size
        self.loader = loader

        self.load_stream = torch.cuda.Stream(device=device)
        self.queue = queue.Queue(maxsize=self.queue_size)

        self.idx = 0
        self.worker = threading.Thread(target=self.load_loop)
        self.worker.setDaemon(True)
        self.worker.start()

    def load_loop(self):
        """ 不断的将cuda数据加载到队列里 """
        # The loop that will load into the queue in the background
        while True:
            for i, sample in enumerate(self.loader):
                self.queue.put(self.load_instance(sample))

    def load_instance(self, sample):
        """ 将batch数据从CPU加载到GPU中 """
        if torch.is_tensor(sample):
            with torch.cuda.stream(self.load_stream):
                return sample.to(self.device, non_blocking=True)
        elif sample is None or type(sample) in (list, str):
            return sample
        elif isinstance(sample, dict):
            return {k: self.load_instance(v) for k, v in sample.items()}
        else:
            return [self.load_instance(s) for s in sample]

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        # 加载线程挂了
        if not self.worker.is_alive() and self.queue.empty():
            self.idx = 0
            self.queue.join()
            self.worker.join()
            raise StopIteration
        # 一个epoch加载完了
        elif self.idx >= len(self.loader):
            self.idx = 0
            raise StopIteration
        # 下一个batch
        else:
            out = self.queue.get()
            self.queue.task_done()
            self.idx += 1
        return out

    def __len__(self):
        return len(self.loader)

    @property
    def sampler(self):
        return self.loader.sampler

    @property
    def dataset(self):
        return self.loader.dataset


class _RepeatSampler(object):
    """ 一直repeat的sampler """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):
    """ 多epoch训练时，DataLoader对象不用重新建立线程和batch_sampler对象，以节约每个epoch的初始化时间 """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
            

data_set = None
train_model = None
loader = MultiEpochsDataLoader(data_set, batch_size=128, 
                               shuffle=True, num_workers=32, 
                               pin_memory=True)
if torch.cuda.is_available():
    loader = CudaDataLoader(loader, 'cuda')

for image in loader:
    # image = image.cuda(non_blocking=True) # 这一步不用做了，做了也没事，几乎不耗时间
    train_model(image)
    
# 数据预处理耗费太多时间的话，可以考虑使用NVIDIA/DALI:https://github.com/NVIDIA/DALI