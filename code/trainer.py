import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.timer import Timer
from base.base_trainer import BaseTrainer
from torch.utils.tensorboard import SummaryWriter

class Trainer(BaseTrainer):
    def __init__(self, loss_name='bce', optimizer_name='adam', lr=1e-4, n_epochs=20, lr_milestones=[],
                 batch_size=128, validate=False, exp_name='experiment', device='cuda', **kwargs):
        super().__init__(loss_name, optimizer_name, lr, n_epochs, lr_milestones, batch_size, device)
        self.validate = validate
        self.exp_name = exp_name

    def train(self, net, train_data, val_data, tensor_board=False):
        train_loss, train_acc = [], []
        loss_epoch, acc_epoch = 0., 0.

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

        if tensor_board:
            writer = SummaryWriter(f'../logs/{self.exp_name}')

        net = net.to(self.device)
        with Timer():
            net.train()
            for n_epoch in range(self.n_epochs):
                loss_epoch = 0.
                acc_epoch = 0.
                val_loss_epoch = 0.
                val_acc_epoch = 0.
                # n_batches = 0
                
                t = tqdm(train_loader)
                batch_no = 0
                for (data_batch, label_batch) in t:        
                    optim.zero_grad()
                    d = data_batch.to(self.device)
                    l = label_batch.to(self.device)
                    pred = net(d)
                    loss = loss_fn(pred, l)

                    # train_loss.append(loss.item())
                    loss_epoch += loss.item()
                    
                    loss.backward()
                    optim.step()
                    scheduler.step()

                    cl = pred.argmax(axis=-1)
                    acc = (cl == l).float().mean()
                    acc_epoch += acc

                    # train_acc.append(acc)
                    if tensor_board:
                        writer.add_scalar('Loss/batch_train', loss, n_epoch*self.batch_size+batch_no)
                        writer.add_scalar('Acc/batch_train', acc, n_epoch*self.batch_size+batch_no)

                    t.set_description(f'Epoch {n_epoch} Loss: {loss:.8f} Acc: {acc:.8f}')
                    t.refresh()
                    batch_no += 1
                loss_epoch /= batch_no
                acc_epoch /= batch_no
                print(f'\n[TRAIN]Epoch {n_epoch} Loss: {loss_epoch} Acc: {acc_epoch}')
                if tensor_board:
                    writer.add_scalar('Loss/train', loss_epoch, n_epoch)
                    writer.add_scalar('Acc/train', acc_epoch, n_epoch)

                if self.validate:
                    val_loss_epoch, val_acc_epoch = self.evaluate(net, val_data, self.loss_name)
                    print(f'\n[VAL]Epoch {n_epoch} Loss: {val_loss_epoch} Acc: {val_acc_epoch}')
                    if tensor_board:
                        writer.add_scalar('Loss/val', val_loss_epoch, n_epoch)
                        writer.add_scalar('Acc/val', val_acc_epoch, n_epoch)
                print()
        if tensor_board:
            writer.close()

    def evaluate(self, net, data, loss_name):
        val_loss, val_acc = [], []
        val_loss_epoch, val_acc_epoch = 0., 0.
        val_loader = DataLoader(data, batch_size=self.batch_size, shuffle=True)

        if loss_name == 'bce':
            loss_fn = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f'Invalid loss name: {self.loss_name}!')

        net.to(self.device)
        net.eval()
        t = tqdm(val_loader)
        with torch.no_grad():
            batch_no = 0
            for (data_batch, label_batch) in t:
                d = data_batch.to(self.device)
                l = label_batch.to(self.device)
                pred = net(d)
                loss = loss_fn(pred, l)

                # val_loss.append(loss.item())
                val_loss_epoch += loss.item()

                cl = pred.argmax(axis=-1)
                acc = (cl == l).float().mean()
                val_acc_epoch += acc

                # val_acc.append(acc)
                t.refresh()
                batch_no += 1
            val_loss_epoch /= batch_no
            val_acc_epoch /= batch_no
        
        return val_loss_epoch, val_acc_epoch
