import os
import torch
import shutil
from torch.optim import Adam

from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

# --------------------------------
# helper functions
# --------------------------------


def check_dir(save_dirs):
    """
    creates the directory to save model
    """
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)


def get_dataloader(dataset, batch_size, data_split_ratio=None, seed=100):
    """
    splits the dataset into train/val/test loaders
    ---------------
    Args
    dataset: pytorch-geometric Dataset generated after parsing
    batch_size (int)
    data_split_ratio (list): training, validation and testing ratio
    seed: random seed to split the dataset randomly
    ---------------
    Returns
    a dictionary of training, validation, and testing dataLoader
    """

    num_train = int(data_split_ratio[0] * len(dataset))
    num_eval = int(data_split_ratio[1] * len(dataset))
    num_test = len(dataset) - num_train - num_eval
    
    train, _eval, test = random_split(dataset,
                                      lengths=[num_train, num_eval, num_test],
                                      generator=torch.Generator().manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(_eval, batch_size=batch_size, shuffle=False)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)
    return dataloader

# -------------------------------
# main trainer class
# -------------------------------


class TrainModel(object):

    """
    classs for trainging GCN/GNN model
    current code requires loading complete dataset to memory (default option for many models)
    if this is inneficient (on RAM), must change code to load only a mini-batch from file
    however this would mean the training time is longer
    """
    
    def __init__(self, model, dataset, device, save_dir=None, save_name='model', ** kwargs):
        """
        Args
        model (torch.model): the GNN model
        device (str): set to cpu or gpu:0
        save_dir (str): apth to save model
        save_name (str): model name
        ** kwargs: will be trainer_params, optimizer_params and dataloader_params
        """
        self.model = model
        self.dataset = dataset  # the loaded dataset from the parser
        self.device = device

        self.optimizer = None
        self.save = save_dir is not None
        self.save_dir = save_dir
        self.save_name = save_name
        check_dir(self.save_dir)

        dataloader_params = kwargs.get('dataloader_params')
        self.loader = get_dataloader(dataset, **dataloader_params)

    # ------------------------
    # internal functions used
    # ------------------------
    
    def __loss__(self, logits, labels):
        """
        cross entropy loss for classification
        """
        return F.cross_entropy(logits, labels)

    def _train_batch(self, data, labels):
        """
        runs single forward pass for mini-batch of data
        ---------------
        Args
        data (torch.Data): the data mini-batch
        labels (torch.Data): class-labels
        ---------------
        Returns
        loss (torch.tensor)
        """
        # 1. forward pass
        logits = self.model(data=data)

        # 2. get loss
        loss = self.__loss__(logits, labels)

        # 3. optimize weights
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=2.0)
        self.optimizer.step()
        
        return loss.item()

    def _eval_batch(self, data, labels, **kwargs):
        """
        similar to _train_batch above, but has no optimizer step
        ---------------
        Args
        data (torch.Data): the data mini-batch
        labels (torch.Data): class-labels
        ---------------
        Returns
        loss (torch.tensor)
        preds (torch.tensor)
        """
        self.model.eval()
        logits = self.model(data)
        loss = self.__loss__(logits, labels)
        loss = loss.item()
        preds = logits.argmax(-1)
        return loss, preds

    # -------------------------
    # main functions
    # -------------------------
    
    def eval(self):
        """
        runs the _eval_batch on eval-dataset
        """
        self.model.to(self.device)
        self.model.eval()  # stops gradient computation

        with torch.no_grad():
            losses, accs = [], []
            for batch in self.loader['eval']:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                accs.append(batch_preds == batch.y)
            eval_loss = torch.tensor(losses).mean().item()
            eval_acc = torch.cat(accs, dim=-1).float().mean().item()

        self.model.train()  # reset gradient computation
        return eval_loss, eval_acc

    def test(self):
        """
        runs _eval_batch() on test-dataset
        """
        state_dict = torch.load(os.path.join(self.save_dir, f'{self.save_name}_best.pth'))['net']
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        with torch.no_grad():
            losses, preds, accs = [], [], []
            for batch in self.loader['test']:
                batch = batch.to(self.device)
                loss, batch_preds = self._eval_batch(batch, batch.y)
                losses.append(loss)
                preds.append(batch_preds)
                accs.append(batch_preds == batch.y)
            test_loss = torch.tensor(losses).mean().item()
            preds = torch.cat(preds, dim=-1)
            test_acc = torch.cat(accs, dim=-1).float().mean().item()
            print(f"Test loss: {test_loss:.4f}, test acc {test_acc:.4f}")

        self.model.train()  # reset gradianet computationn
        return test_loss, test_acc, preds

    def train(self, train_params=None, optimizer_params=None):
        """
        runs training iteration
        ------------------
        Args
        train_params (dict): the train_params set when initializing the trainer class
        optimizer_params (dict): the optimizer params set when initializing the trainer class
        """
        num_epochs = train_params['num_epochs']
        num_early_stop = train_params['num_early_stop']
        milestones = train_params['milestones']
        gamma = train_params['gamma']

        # intialize optimizer
        if optimizer_params is None:
            self.optimizer = Adam(self.model.parameters())
        else:
            self.optimizer = Adam(self.model.parameters(), **optimizer_params)

        # set the learning rate, fixed or as a schedule
        if milestones is not None and gamma is not None:
            lr_schedule = MultiStepLR(self.optimizer,
                                      milestones=milestones,
                                      gamma=gamma)
        else:
            lr_schedule = None

        # train model
        self.model.to(self.device)
        best_eval_acc = 0.0
        best_eval_loss = 0.0
        early_stop_counter = 0

        # loop through for all epochs
        for epoch in range(num_epochs):
            is_best = False
            self.model.train()
            losses = []

            # for each epoch, loop through all batches
            for idx, batch in enumerate(self.loader['train']):
                batch = batch.to(self.device)
                loss = self._train_batch(batch, batch.y)
                losses.append(loss)

            # compute loss and stop early if needed
            train_loss = torch.FloatTensor(losses).mean().item()
            eval_loss, eval_acc = self.eval()
            print(f'Epoch:{epoch}, Training_loss:{train_loss:.4f}, Eval_loss:{eval_loss:.4f}, Eval_acc:{eval_acc:.4f}')
            if num_early_stop > 0:
                if eval_loss <= best_eval_loss:
                    best_eval_loss = eval_loss
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                if epoch > num_epochs / 2 and early_stop_counter > num_early_stop:
                    break

            # change learning rate if there is a schedule
            if lr_schedule:
                lr_schedule.step()

            if best_eval_acc < eval_acc:
                is_best = True
                best_eval_acc = eval_acc
            recording = {'epoch': epoch, 'is_best': str(is_best)}
            if self.save:
                self.save_model(is_best, recording=recording)

    def save_model(self, is_best=False, recording=None):
        """
        saves model to file, defaults to save best version based on eval.
        """
        self.model.to('cpu')
        state = {'net': self.model.state_dict()}
        for key, value in recording.items():
            state[key] = value
        latest_pth_name = f"{self.save_name}_latest.pth"
        best_pth_name = f'{self.save_name}_best.pth'
        ckpt_path = os.path.join(self.save_dir, latest_pth_name)
        torch.save(state, ckpt_path)
        if is_best:
            print('saving best model version...')
            shutil.copy(ckpt_path, os.path.join(self.save_dir, best_pth_name))
        self.model.to(self.device)

    def load_model(self):
        """
        loads model from file
        """
        state_dict = torch.load(os.path.join(self.save_dir, f"{self.save_name}_best.pth"))['net']
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
