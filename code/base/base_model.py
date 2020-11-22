from abc import ABC, abstractclassmethod
from shutil import Error
from trainer import Trainer
import torch

class BaseModel(object):
    def __init__(self, device='cuda', **kwargs):
        super().__init__()
        self.device = device
        self.net = None
        self.trainer = None

    def train(self, train_data, val_data=None, loss_name='bce', optimizer_name='adam', lr=1e-4,
              n_epochs=20, lr_milestones=[], batch_size=128, tensor_board=False,
              device='cuda', **kwargs):
        self.trainer = Trainer(loss_name, optimizer_name, lr, n_epochs, lr_milestones, batch_size,
            val_data is not None, device, **kwargs)
        
        self.trainer.train(self.net, train_data, val_data, tensor_board)

        print(f'Training finished!')
    
    def test(self, test_data, loss_name='bce'):
        if self.train is None:
            raise Error('Test called before train!')

        test_loss, test_acc = self.trainer.evaluate(self.net, test_data, loss_name)

        print(f'\n[TEST] Loss: {test_loss} Acc: {test_acc}')
    
    def save_model(self, path):
        assert self.net is not None, 'First initialize/train your model!'
        torch.save(self.model.net.state_dict(), path)
        print(f'Model saved to {path}')
    
    def load_model(self, path):
        assert self.net is not None, 'First initialize your model!'
        self.net.load_state_dict(torch.load(path))
        print(f'Model load from {path}')

    # def predict(self, data, batch_size=None):
    #     loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False)
    #     self.net.eval()

    #     with torch.no_grad():
    #         predictions = torch.tensor([], device=self.device, dtype=torch.double)

    #         if batch_size is None:
    #             predictions = self.net(data.to(self.device))
    #         else:
    #             for data_batch in loader:
    #                 predictions = torch.cat([predictions, self.net(data_batch.to(self.device))])
            
    #     return predictions