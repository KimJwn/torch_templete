import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
    
def Accuracy(output, targets):
    preds = output.argmax(dim=1)    # 가장 높은 값을 가진 인덱스를 출력한다. 
    return int( preds.eq(targets).sum() )

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def calculate_norm(dataset):
    # dataset의 axis=1, 2에 대한 평균 산출
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 평균 산출
    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    # dataset의 axis=1, 2에 대한 표준편차 산출
    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])
    # r, g, b 채널에 대한 각각의 표준편차 산출
    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()
    
    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

def create_save_dir(name, default_dir = 'result'):
    try:
        check_mkdir(default_dir)
        check_mkdir(default_dir+'/'+name)
    except:
        print("Error : Failed to create the directory")
    return default_dir +'/'+name+'/'
#
def check_mkdir(dir_name):
    if not os.path.exists(dir_name): os.mkdir(dir_name)
    return dir_name
#
def get_logger(path, mode='train'):
    logger = logging.getLogger()
    if len(logger.handlers) > 0 : return logger
    
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    file_handler = logging.FileHandler(os.path.join(path, mode+'.log' )) # 'train.log'
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
import tqdm
import logging
import numpy as np
import gc

''' CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=gpu 04_templete.py 4 2 '''

MD_PTH = '/media/data2/jiwon/'

'''*Newrly Updated For Torchrun Multi GPU DDP*'''
def ddp_setup():
    init_process_group(backend="nccl")

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        save_every: int,
        snapshot_path: str, #'''*Newrly Updated For Torchrun Multi GPU DDP*'''
        pj_name: str,
        valid_data = False
    ) -> None:
        '''*Newrly Updated For Torchrun Multi GPU DDP*'''
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.dataloader = {'train': train_data}
        if valid_data != False :
            self.dataloader['valid'] = valid_data
        self.optimizer = optimizer
        self.criterion = loss_fn
        self.save_every = save_every
        
        '''*Newrly Updated For Torchrun Multi GPU DDP*'''
        self.epochs_run = 0
        PJ_PTH = check_mkdir(os.path.join(MD_PTH, pj_name))
        SAVE_PTH = check_mkdir( os.path.join(PJ_PTH, 'save'))
        if self.gpu_id == 0:    
            self.logger = get_logger(SAVE_PTH)
                
        
        self.snapshot_path = os.path.join(SAVE_PTH, snapshot_path)
        if os.path.exists(self.snapshot_path):  self._load_snapshot()

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.mode = 'train'
        


    '''*Newrly Updated For Torchrun Multi GPU DDP*'''
    def _load_snapshot(self):
        self.printonce("Loading snapshot")
        loc = f"cuda:{self.gpu_id}"
        
        snapshot = torch.load(self.snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        
        self.printonce(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(), Accuracy(output, targets)

    def _run_epoch(self, epoch, log):
        b_sz = len(next(iter(self.dataloader.get(self.mode)))[0])
        losses, accuracies = AverageMeter(), AverageMeter()
        
        self.dataloader.get(self.mode).sampler.set_epoch(epoch)
        
        tq = tqdm.tqdm(total=len(self.dataloader.get(self.mode))*b_sz)
        tq.set_description(f"[GPU {self.gpu_id} - {self.mode}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.dataloader.get(self.mode))}")
        for b_idx, (source, targets) in enumerate(self.dataloader.get(self.mode)):
            mini_b_sz = source.shape[0]
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss, acc = self._run_batch(source, targets)
            losses.update(loss, mini_b_sz)
            accuracies.update(acc, mini_b_sz)
            tq.set_postfix(loss='{:.5f}'.format(losses.avg), acc='{:.5f}'.format(accuracies.avg))
            tq.update(b_sz)
        tq.close()
        log[f'{self.mode} loss'] = losses.avg
        log[f'{self.mode} accuracy'] = accuracies.avg
        return log

    '''*Newrly Updated For Torchrun Multi GPU DDP*'''
    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, os.path.join(self.snapshot_path))
        # self.printonce(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    '''*Newrly Updated For Torchrun Multi GPU DDP*'''
    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            log = {'epoch': epoch}
            self.printonce('\n')
            #train
            self.mode = 'train'
            self.model.train()
            log = self._run_epoch(epoch, log)
            
            if self.valid_data != False:
                #valid
                self.mode = 'valid'
                self.model.eval()
                log = self._run_epoch(epoch, log)
                
            if self.gpu_id == 0 :
                self.logger.info(log)
                if epoch % self.save_every == 0:    self._save_snapshot(epoch)
            
        gc.collect()
        torch.cuda.empty_cache()

    def printonce(self,message:str):
        if self.gpu_id == 0: print(message)
            
def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = torch.nn.Linear(20, 1)  # load your model
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def split_set(dataset: Dataset, scale=1, prop=.8):
    origin_sz = int(len(dataset))
    use_sz = int(origin_sz* scale)
    if scale < 1 : dataset, _ = random_split(dataset, [use_sz, origin_sz-use_sz])
    print(int(use_sz*prop), use_sz-int(use_sz*prop))
    train, test = random_split(dataset, [int(use_sz*prop), use_sz-int(use_sz*prop)])
    return train, test

def prepare_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

'''*Newrly Updated For Torchrun Multi GPU DDP*'''
def main(save_every: int, total_epochs: int, batch_size: int, pj_name:str, snapshot_path: str = "snapshot.pt"):
    gc.collect()
    torch.cuda.empty_cache()
    
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_set, valid_set =  split_set(dataset, scale=1, prop=.8)
    train_loader = prepare_dataloader(train_set, batch_size)
    valid_loader = prepare_dataloader(valid_set, batch_size)
    trainer = Trainer(model, train_loader, optimizer, F.cross_entropy, save_every, snapshot_path, pj_name, valid_loader)
    trainer.train(total_epochs)
    destroy_process_group()


'''*Newrly Updated For Torchrun Multi GPU DDP*'''
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--pj_name', '-n',  type=str, default='templete', help='What is your project name')
    parser.add_argument('--batch_size', '-b', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    
    main(args.save_every, args.total_epochs, args.batch_size, args.pj_name)