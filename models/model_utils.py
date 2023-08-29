#%% import
import os
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,random_split,DataLoader
from torch.utils.tensorboard import SummaryWriter

#%% process
def train_step(model,x,y):
    model.train()
    
    model.optimizer.zero_grad()
    y_pred = model(x)
    loss = model.loss_func(y_pred,y)
    with torch.no_grad():
        metric = model.metric_func(y_pred,y)

    # with torch.autograd.detect_anomaly():
    #     loss.backward()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),max_norm=10.0,norm_type=2)
    model.optimizer.step()
    
    return loss.cpu().item(),metric

def validate_step(model,x,y):
    model.eval()
    
    y_pred = model(x)
    loss = model.loss_func(y_pred,y)
    with torch.no_grad():
        metric = model.metric_func(y_pred,y)
    
    return loss.cpu().item(),metric

def resume(model):
    # 断点续训：寻找并导入checkpoint
    saved_step = 0
    saved_file = None
    current_step = 0
    
    global_epoch = 1
    global_step = 1
    global_loss = 0.0
    
    model_dir = model.model_dir
    if os.path.isdir(model_dir+'checkpoints/'):
        checkpoints = os.listdir(model_dir+'checkpoints/')
        if len(checkpoints)>0:
            for file in checkpoints:
                if file.startswith('checkpoint'):
                    tokens = file.split('.')[0].split('-')
                    if len(tokens) != 2:
                        continue
                    current_step = int(tokens[1])
                    if current_step >= saved_step:
                        saved_file = file
                        saved_step = current_step
            checkpoint_path = model_dir + 'checkpoints/' + saved_file
            checkpoint = torch.load(checkpoint_path)

            model.load_state_dict(checkpoint['model'])
            global_epoch = checkpoint['epoch']+1  # 设置开始的epoch
            global_step = checkpoint['step']+1  # 设置开始的step
            global_loss = checkpoint['loss']
        else:
            print("No exisiting model !")
    else:
        print("No exisiting model !")

    return global_epoch,global_step,global_loss

def check(model,epoch,step,loss):
    # 断点续训，保存checkpoint
    model_dir = model.model_dir
    checkpoint = {"model":model.state_dict(),"epoch":epoch,"step":step,"loss":loss}
    if not os.path.isdir(model_dir+'checkpoints/'):
        os.mkdir(model_dir+'checkpoints/')
    torch.save(checkpoint,model_dir+'checkpoints/'+'checkpoint-%s.pth' % (str(step)))

def train(model,epochs,train_dl,val_dl,config):
    model.train()
    model_dir = model.model_dir
    # 断点续训：寻找并导入checkpoint，正在处理的epoch和step，以及对应loss
    global_epoch,global_step,global_loss = resume(model)
    model.to(config['device'])
    
    train_log_step_freq = 10
    val_step_freq = 50
    check_step_freq = 100
    
    # tensorboard:初始化，若路径不存在会创建路径
    writer_step = SummaryWriter(log_dir=model_dir+'tb-step-logs',purge_step=global_step)
    # writer_epoch = SummaryWriter(log_dir=model_dir+'tb-epoch-logs',purge_step=global_epoch)
    
    wandb.init(project='surrogate-model',job_type='test',group=config['model_name'],id='666',
               config=config,save_code=True,resume='allow')
    # artifact = wandb.Artifact(name='train_script',type='code')
    # artifact.add_file('/models/model_utils.py')
    wandb.watch(models=model,log='all',log_freq=200,log_graph=True)
    
    # region hook
    # def tb_hook(name):
    #     def hook(module,input,output):
    #         if isinstance(module,(nn.ReLU)):
    #             try:
    #                 writer_step.add_histogram(name+'/output',output[0],global_step=global_step)
    #             except:
    #                 pass
    #     return hook
    
    # for name,module in model.named_modules():
    #     module.register_forward_hook(tb_hook(name))
    # endregion
    
    # train loop
    loss_sum = global_loss
    for _ in range(epochs):
        pbar = tqdm(total=len(train_dl),desc=f"training epoch {global_epoch}")
        # train
        for _,(x,y) in enumerate(train_dl):
            loss,_ = train_step(model,x,y)
            loss_sum += loss
            pbar.update(1)
            
            if global_step % train_log_step_freq == 0:
                writer_step.add_scalars('loss',{'train':loss_sum/train_log_step_freq},
                                        global_step=global_step)
                wandb.log({'train_loss':loss_sum/train_log_step_freq},step=global_step)
                loss_sum = 0.0
            
            if global_step % val_step_freq == 0:
                torch.cuda.empty_cache()  # 开始验证，清理一下缓存
                # validate
                loss_sum_val = 0.0
                metric_sum_val = {metric:0.0 for metric in model.metrics}
                
                for step_val,(x,y) in enumerate(val_dl,1):
                    with torch.no_grad():
                        loss_val,metric_val = validate_step(model,x,y)
                    loss_sum_val += loss_val
                    for metric in model.metrics:
                        metric_sum_val[metric] += metric_val[metric]
               
                for metric in model.metrics:
                    output_list = [str(i)+'-step-ahead' for i in range(1,model.lookahead+1)]
                    output_list += [str(i) for i in range(12)]
                    writer_step.add_scalars(metric,{key:value/step_val for (key,value) in zip(output_list,metric_sum_val[metric])},
                                            global_step=global_step)
                    wandb.log({(metric+':'+key):value/step_val for (key,value) in zip(output_list,metric_sum_val[metric])},step=global_step)
                writer_step.add_scalars('loss',{'val':loss_sum_val/step_val},global_step=global_step)
                wandb.log({'val_loss':loss_sum_val/step_val},step=global_step)

            # 断点续训，steps级checkpoint
            if global_step % check_step_freq == 0:
                check(model,global_epoch,global_step,loss_sum)
            global_step += 1
        
        pbar.close()
        # 断点续训，epochs级checkpoint
        if global_epoch % 1 == 0:
            check(model,global_epoch,global_step-1,loss_sum)
        global_epoch += 1
        
        model.scheduler.step()  # 随epoch衰减
        
    wandb.finish()

#%% data
class MyDataset(Dataset):
    def __init__(self,config):
        super(MyDataset,self).__init__()
        self.config = config
        self.data_dir = self.config['data_dir']
        self.lookback = self.config['lookback']  # 从数据中选取的时窗大小
        self.lookahead = self.config['lookahead']  # 从数据中选取的时窗大小
        self.sample_size = self.config['sample_size']
        
        assert self.lookback <= self.sample_size[0], "lookback out of range"
        assert self.lookahead <= self.sample_size[1], "lookahead out of range"
        with open(self.data_dir+'sample2chunk.pkl','rb') as f:
            self.sample2chunk = joblib.load(f)
        
    def __getitem__(self,index):
        # 在这里把数据方法device上
        chunk_index,sample_index = self.sample2chunk[index]
        
        other = torch.load(self.data_dir+f'other_{chunk_index}_{sample_index}.pth')  # other: (delay,delay_m,tc)
        obs = np.empty(self.lookback,dtype=object)
        # obs: (lookahead,frames!,***)
        for i,v in enumerate(np.load(self.data_dir+f'obs_{chunk_index}_{sample_index}.npz',allow_pickle=True)['arr_0']):
            obs[i] = v  # 别动，内存不够
        
        obs = obs[None,(self.sample_size[0]-self.lookback):self.sample_size[0]]
        
        tc = other['tc'][None,(self.sample_size[0]-self.lookback):(self.sample_size[0]+self.lookahead)]  # tc:(1,lookback + lookahead,*)
        x = {'obs':obs,'tc':tc}
        y = other['timeloss'][None,:self.lookahead]  # (1,lookahead), 数据可以更长但只取lookahead那么长
        
        return (x,y)
        
    def __len__(self):
        return len(self.sample2chunk)

def train_val_test_split(dataset,splits,seed):
    train_size = int(splits[0]*len(dataset))
    val_size = int(splits[1]*len(dataset))
    test_size = len(dataset)-train_size-val_size

    # 随机数种子需要设置
    train_ds,val_ds,test_ds = random_split(dataset,[train_size,val_size,test_size],
                                           generator=torch.Generator().manual_seed(seed))

    return train_ds,val_ds,test_ds

def my_collate_fn(samples):
    x = {}
    x['obs'] = np.concatenate([sample[0]['obs'] for sample in samples],axis=0)
    x['tc'] = torch.cat([sample[0]['tc'] for sample in samples],dim=0)
    
    y = torch.cat([sample[1] for sample in samples],dim=0)
    
    return (x,y)

def get_dataloader(dataset,config):
    train_ds,val_ds,test_ds = train_val_test_split(dataset,config['split'],seed=config['seed'])
    
    # dataloader使用固定的随机数种子
    train_dl = DataLoader(train_ds,batch_size=config['batch_size'],collate_fn=my_collate_fn,
                          shuffle=True,drop_last=True,generator=torch.Generator().manual_seed(config['seed']),
                          num_workers=config['num_workers'])
    val_dl = DataLoader(val_ds,batch_size=config['batch_size'],collate_fn=my_collate_fn,
                        shuffle=False,drop_last=True,generator=torch.Generator().manual_seed(config['seed']),
                        num_workers=config['num_workers'])
    test_dl = DataLoader(test_ds,batch_size=config['batch_size'],collate_fn=my_collate_fn,
                         shuffle=False,drop_last=True,generator=torch.Generator().manual_seed(config['seed']),
                         num_workers=1)
    # 为严格保证batch_size，设置drop_last=True
    return train_dl,val_dl,test_dl
