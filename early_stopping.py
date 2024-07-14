import torch
import numpy as np

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False

    def __call__(self, val_loss, model, save_path):

        if val_loss > self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # torch.save(model, save_path + "/" +
            #            "out_checkpoint_{:.6f}.pth".format(val_loss))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save_checkpoint(val_loss, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_path):
        '''Saves model when validation loss decrease.'''
        if torch.distributed.get_rank() == 0:
            if self.verbose:
                print(f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model, save_path + "/" + "checkpoint_{:.6f}.pth".format(val_loss))
            self.best_score = val_loss