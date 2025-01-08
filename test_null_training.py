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
def get_loss(predictions, batch_output, debug_print=False):
    training_policies = batch_output[:,1:]
    policy_masks = (training_policies == 0)
    loss_value = loss_fn_value(predictions[:,:1], batch_output[:,:1])
    predicted_policy = predictions[:,1:].masked_fill(policy_masks, -1e9)
    loss_policy = F.cross_entropy(predicted_policy, training_policies)
    if debug_print:
        print(f'value loss: {loss_value}, policy loss: {loss_policy}')
    return loss_value + loss_policy * loss_policy_weight
    
def test_null_training():
    net = NullTrainingNetwork().cuda()
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'Created network with {pytorch_total_params} total parameters.')
    print('Loading dataset...')
    nullset = torch.load('nullset_100K.tnsr', weights_only=False)
    X = nullset['X'].cuda()
    Y = nullset['Y'].cuda()
    print(f'Loaded dataset. Inputs are of shape {X.shape}, outputs of shape {Y.shape}.')
    split = math.floor(len(X) * 0.9)
    X_train = X[:split,:]
    Y_train = Y[:split,:]
    training_batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)
    X_test = X[split:,:]
    Y_test = Y[split:,:]

    # Measure starting loss.
    net.eval()
    loss = get_loss(net(X_test), Y_test, debug_print=True)
    print(f'Starting test loss: {loss}.')
    # Train.
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(1000):
        print(f'Starting epoch {epoch + 1} of 1000.')
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
        train_loss = get_loss(net(X_train), Y_train, debug_print=False)
        test_loss = get_loss(net(X_test), Y_test, debug_print=True)
        print(f'Finished in {elapsed_time:.1f}s. Train loss: {train_loss}. Test loss: {test_loss}.')

test_null_training()