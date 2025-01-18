from os import listdir
import torch

@torch.no_grad()
def combine_all_selfplay():
    Xs = []
    Ys = []
    for file in listdir('alphapush-selfplay-results'):
        chunk = torch.load(f'alphapush-selfplay-results/{file}', weights_only=False)
        Xs.append(chunk['X'])
        Ys.append(chunk['Y'])
    X = torch.cat(Xs, dim=0)
    Y = torch.cat(Ys, dim=0)
    print(X.shape, Y.shape)
    torch.save({'X': X, 'Y': Y}, 'dataset_270K_v007_2470K.tnsr')

combine_all_selfplay()