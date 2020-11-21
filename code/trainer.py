import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.timer import Timer
from base.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, loss_name='bce', optimizer_name='adam', lr=1e-4, n_epochs=20, lr_milestones=[],
                 batch_size=128, validate=False, device='cuda', **kwargs):
        super().__init__(loss_name, optimizer_name, lr, n_epochs, lr_milestones, batch_size, device)
        self.validate = validate

    def train(self, net, train_data, val_data):
        train_loss, val_loss = [], []
        train_acc, val_acc = [], []
        loss_epoch, val_loss_epoch = 0., 0.
        acc_epoch, val_acc_epoch = 0., 0.

        if self.loss_name == 'bce':
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Invalid loss name: {self.loss_name}!')

        if self.optimizer_name == 'adam':
            optim = torch.optim.Adam(net.parameters(), lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optim = torch.optim.SGD(net.parameters(), lr=self.lr)
        elif self.optimizer_name == 'rms':
            optim = torch.optim.RMSprop(net.parameters(), lr=self.lr)
        else:
            raise ValueError(f'Invalid optimizer name: {self.optimizer_name}!')
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=self.lr_milestones, gamma=0.1)

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        if self.validate:
            val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        net = net.to(self.device)
        with Timer():
            net.train()
            print(f'Optimization started!')
            for n_epoch in range(self.n_epochs):
                loss_epoch = 0.
                acc_epoch = 0.
                val_loss_epoch = 0.
                val_acc_epoch = 0.
                # n_batches = 0
                
                t = tqdm(train_loader)
                for batch_no, (data_batch, label_batch) in enumerate(t):
                    
                    d = data_batch.to(self.device)
                    l = label_batch.to(self.device)
            
                    optim.zero_grad()
                    pred = net(d)
                    loss = loss_fn(pred, l)

                    train_loss.append(loss.item())
                    loss_epoch += loss.item()
                    
                    loss.backward()
                    optim.step()
                    scheduler.step()

                    cl = pred.argmax(axis=-1)
                    acc = (cl == l).float().mean()
                    acc_epoch += acc

                    train_acc.append(acc)

                    t.set_description(f'Epoch {n_epoch} Loss: {loss:.8f} Acc: {acc:.8f}')
                    t.refresh()
                loss_epoch /= batch_no
                acc_epoch /= batch_no
                print(f'\n[TRAIN]Epoch {n_epoch} Loss: {loss_epoch} Acc: {acc_epoch}')

                if self.validate:
                    net.eval()
                    t = tqdm(val_loader)
                    for batch_no, (data_batch, label_batch) in enumerate(t):
                        d = data_batch.to(self.device)

                        l = label_batch.to(self.device)
                        pred = net(d)
                        
                        loss = loss_fn(pred, l)
                        val_loss.append(loss.item())
                        val_loss_epoch += loss.item()

                        cl = pred.argmax(axis=-1)
                        acc = (cl == l).float().mean()
                        val_acc_epoch += acc

                        val_acc.append(acc)

                        t.set_description(f'Validation Loss: {loss:.8f} Acc: {acc:.8f}')
                        t.refresh()
                    net.train()
                    val_loss_epoch /= batch_no
                    val_acc_epoch /= batch_no
                    print(f'[VAL]Epoch {n_epoch} Loss: {val_loss_epoch} Acc: {val_acc_epoch}')
