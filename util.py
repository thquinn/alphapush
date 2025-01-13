from os import listdir
import torch

@torch.no_grad()
def combine_sets():
    nullset1 = torch.load('dataset_270K_v000_10K.tnsr', weights_only=False)
    X1 = nullset1['X']
    Y1 = nullset1['Y']
    nullset2 = torch.load('dataset_270K_v000_10K2.tnsr', weights_only=False)
    X2 = nullset2['X']
    Y2 = nullset2['Y']
    nullset3 = torch.load('dataset_270K_v000_1736414033.tnsr', weights_only=False)
    X3 = nullset3['X']
    Y3 = nullset3['Y']
    X = torch.cat([X1, X2, X3], dim=0)
    Y = torch.cat([Y1, Y2, Y3], dim=0)
    print(X.shape, Y.shape)
    torch.save({'X': X, 'Y': Y}, 'dataset_270K_v000_130K.tnsr')

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
    torch.save({'X': X, 'Y': Y}, 'dataset_270K_v003_640K.tnsr')

combine_all_selfplay()