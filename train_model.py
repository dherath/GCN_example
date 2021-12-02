
import torch
from code.dataprocessor_bamotif import example_BAMOTIF


def main():
    print('+ started GCN + BA2Motif example...')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    filename = "./datasets/BA_2Motifs.pkl"

    # dataprocesssor class, will need to modify this for other data
    dataprocessor = example_BAMOTIF()
    dataset = dataprocessor.parse_all_data(filename)
    print('+ data BA2Motif loaded')

    
    return


if __name__ == "__main__":
    main()
