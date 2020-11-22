from abc import ABC, abstractclassmethod, abstractmethod

class BaseTrainer(ABC):
    def __init__(self, loss_name, optimizer_name, lr, n_epochs, lr_milestones, batch_size,
                 device):
        self.loss_name = loss_name
        self.optimizer_name=optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size


        self.device = device
    
    @abstractmethod
    def train(self, net, data, val_data, tensor_board):
        pass

    @abstractclassmethod
    def test(self, net, data, loss_name):
        pass
