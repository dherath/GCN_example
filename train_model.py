import os
import torch
from code.dataprocessor_bamotif import ExampleBAMOTIF
from code.model import GCN
from code.trainer import TrainModel


def main():

    """
    Note: I have not tuned the model to its best hyper-parameter combination for this dataset
    """
    
    print('+ started GCN + BA2Motif example...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    filename = "./datasets/BA_2Motifs.pkl"  # this is a a toy graph classification dataset

    """
    1. dataprocesssor class, will need to modify this
    for other data (can use this as skeleton)
    """

    dataprocessor = ExampleBAMOTIF()
    dataset = dataprocessor.parse_all_data(filename)
    print('+ data BA2Motif loaded')

    """
    2. define the model (currently model is a 3xlayer GCN)
    
    dim_node = the #feats
    dim_hidden = the hidden dimensions per node after graph convolution
    the final size of 16 would be the node embedding size
    num_classes = number of class labels
    dropout_level = the dropout probability
    """

    classifier = GCN(
        dim_node=10,
        dim_hidden=[64, 32, 16],
        num_classes=2,
        dropout_level=0.01
    )
    print('+ classifier model:\n', classifier)

    """
    3. training the model
    3.1. dataloader params

    > Notes
    obtaining train/val/test datasets
    train/val/test split = 80%, 10%, 10%
    """

    dataloader_params = {
        'batch_size': 256,
        'data_split_ratio': [0.8, 0.1, 0.1],
        'seed': 100
    }

    """
    3.2. the trainer params

    > Notes
    can set model to stop early if required (checks for current_eval_loss <= best_eval_loss to stop)
    optionally can set a learning_rate scheduler
    this uses MultiStepLR from torch.optim.lr_scheduler
    the parameters for setting it are milestones and gamma (defults to None)
    current version has a single learning rate throughout all epochs
    """

    train_params = {
        'num_epochs': 20,
        'num_early_stop': 10,
        'milestones': None,
        'gamma': None
    }

    """
    3.3. set params for optimizer

    > Notes
    sets the starting learning rate and weight decay
    """

    optimizer_params = {
        'lr': 0.01,
        'weight_decay': 5e-8
    }

    path = './'  # will use this to generate path to save models

    """
    3.4. create the trainer class
    """

    trainer = TrainModel(
        model=classifier,
        dataset=dataset,
        device=device,
        save_dir=os.path.join(path, 'models'),
        save_name='GCN_Ba2motif_example',
        dataloader_params=dataloader_params
    )
    print('+ trainer class created...\n')

    """
    3.5. run training iterations
    """

    trainer.train(
        train_params=train_params,
        optimizer_params=optimizer_params
    )
    
    """
    3.6. run final test() on model
    """
    
    _, _, _ = trainer.test()
    return


if __name__ == "__main__":
    main()
