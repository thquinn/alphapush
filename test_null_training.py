import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class NullTrainingNetwork(nn.Module):
    def __init__(self):
        super(NullTrainingNetwork, self).__init__()
        input_size = 160
        output_size = 807
        hidden_layers = [input_size, 208, 104, 104, 104, 208]
        self.model = nn.Sequential(
            *[
                item for i in range(len(hidden_layers) - 1)
                for item in (
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_layers[i + 1]),
                    nn.Dropout(0.2),
                )
            ],
            nn.Linear(hidden_layers[-1], output_size),
        )
        self.value_activation = nn.Tanh()
    
    def forward(self, x):
        features = self.model(x)
        value = self.value_activation(features[0:1])
        policy = features[1:]
        return torch.cat((value, policy))

class NullTrainingNetworkV2(nn.Module):
    def __init__(self):
        super(NullTrainingNetworkV2, self).__init__()
        input_size = 160
        output_size = 807
        hidden_layers = [input_size, 416, 208, 208, 208, 208, 208, 208, 208, 208, 416]
        self.model = nn.Sequential(
            *[
                item for i in range(len(hidden_layers) - 1)
                for item in (
                    nn.Linear(hidden_layers[i], hidden_layers[i + 1]),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_layers[i + 1]),
                    nn.Dropout(0.5),
                )
            ],
            nn.Linear(hidden_layers[-1], output_size),
        )
        self.value_activation = nn.Tanh()
    
    def forward(self, x):
        features = self.model(x)
        value = self.value_activation(features[0:1])
        policy = features[1:]
        return torch.cat((value, policy))

loss_fn_value = nn.MSELoss()
loss_policy_weight = 0.25
def get_loss(predictions, batch_output, test_loss=False):
    training_policies = batch_output[:,1:]
    policy_masks = (training_policies == 0)
    loss_value = loss_fn_value(predictions[:,:1], batch_output[:,:1])
    predicted_policy = predictions[:,1:].masked_fill(policy_masks, -1e9)
    loss_policy = F.cross_entropy(predicted_policy, training_policies)
    if test_loss:
        print(f'Test value loss: {loss_value:.4f}, test policy loss: {loss_policy:.4f}.')
    return loss_value + loss_policy * loss_policy_weight
    
def test_null_training():
    # net = NullTrainingNetworkV2().cuda()
    net = torch.load('model_270K_v003.pt', weights_only=False).cpu()
    net.model[19].p = 0.5 # Dropout percentage.
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'Created network with {pytorch_total_params} total parameters.')
    print('Loading dataset...')
    dataset = torch.load('dataset_270K_v002_710K.tnsr', weights_only=False)
    X = dataset['X']
    Y = dataset['Y']
    print(f'Loaded dataset. Inputs are of shape {X.shape}, outputs of shape {Y.shape}.')
    split = math.floor(len(X) * 0.9)
    X_train = X[:split,:].cuda()
    Y_train = Y[:split,:].cuda()
    training_batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=128, shuffle=True, drop_last=True)
    X_test = X[split:,:]
    Y_test = Y[split:,:]

    # Measure starting loss.
    net.eval()
    min_test_loss = get_loss(net(X_test), Y_test, test_loss=True)
    net.cuda()
    print(f'Starting test loss: {min_test_loss:.4f}.')
    # Train.
    optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(1000):
        net.train()
        start_time = time.time()
        for (batch_input, batch_output) in training_batches:
            predictions = net(batch_input)
            loss = get_loss(predictions, batch_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elapsed_time = time.time() - start_time
        net.eval()
        with torch.no_grad():
            X_train_pred = net(X_train)
            train_loss = get_loss(X_train_pred, Y_train, test_loss=False)
            net.cpu()
            X_test_pred = net(X_test)
            test_loss = get_loss(X_test_pred, Y_test, test_loss=True)
            net.cuda()
        torch.cuda.empty_cache()
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(net, 'model_270K_v004.pt')
        print(f'Epoch {epoch + 1} finished in {elapsed_time:.1f}s. Train/test loss: {train_loss:.4f}/{test_loss:.4f}.')

if __name__ == '__main__':
    test_null_training()