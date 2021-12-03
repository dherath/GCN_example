### Example: pytorch-geometric GCN model

To run the code

` python train_model.py
`

The model architecture in this example:

```
GCN(
  (conv1): GCNConv(10, 64)
  (convs): ModuleList(
    (0): GCNConv(64, 32)
    (1): GCNConv(32, 16)
  )
  (relu1): ReLU()
  (relus): ModuleList(
    (0): ReLU()
    (1): ReLU()
  )
  (readout): GlobalMeanPool()
  (ffn): Sequential(
    (0): Dropout(p=0.01, inplace=False)
    (1): Linear(in_features=16, out_features=2, bias=True)
  )
)
```
The model architecture can be changed in **train_model.py**

+ code/
  - dataproccessor_bamotif.py: has the methods to convert the dataset into pytorch-geometric acceptable format
  - model.py: has the definition of the GCN model and the pooling layers
  - trainer.py: has the trainer class, will train/eval/test based on dataset and save model state

+ The current code uses [BA2Motif](https://arxiv.org/pdf/2011.04573.pdf) graph dataset

+ Note: the model in this example has not been tuned
