
import torch
from code.dataprocessor_bamotif import example_BAMOTIF
from code.model import GCN


def main():
    print('+ started GCN + BA2Motif example...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    filename = "./datasets/BA_2Motifs.pkl"

    # 1. dataprocesssor class, will need to modify this for other data (can use this as skeleton)
    dataprocessor = example_BAMOTIF()
    dataset = dataprocessor.parse_all_data(filename)
    print('+ data BA2Motif loaded')

    # 2. define the model
    # dim_node = the #feats
    # dim_hidden = the hidden dimensions per node after graph convolution
    # the final size of 128 would be the node embedding size
    # num_classes = number of class labels
    # dropout_level = the dopout probability
    classifier = GCN(
        dim_node=10,
        dim_hidden=[512, 256, 128],
        num_classes=2,
        dropout_level=0.1
    )
    print('+ classifier model:\n', classifier)

    # 3. training the model
    
    return


if __name__ == "__main__":
    main()
