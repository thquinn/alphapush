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
    # net = NullTrainingNetwork().cuda()
    net = torch.load('model_270K_v002.pt', weights_only=False).cuda()
    net.model[19].p = 0.2 # Dropout percentage.
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'Created network with {pytorch_total_params} total parameters.')
    print('Loading dataset...')
    nullset = torch.load('dataset_270K_v001_300K.tnsr', weights_only=False)
    X1 = nullset['X'].cuda()
    Y1 = nullset['Y'].cuda()
    nullset = torch.load('dataset_270K_v002_470K.tnsr', weights_only=False)
    X2 = nullset['X'].cuda()
    Y2 = nullset['Y'].cuda()
    X = torch.cat([X1, X2], dim=0)
    Y = torch.cat([Y1, Y2], dim=0)
    torch.manual_seed(0)
    randperm = torch.randperm(X.shape[0])
    X = X[randperm].view(X.size())
    Y = Y[randperm].view(Y.size())
    print(f'Loaded dataset. Inputs are of shape {X.shape}, outputs of shape {Y.shape}.')
    split = math.floor(len(X) * 0.9)
    X_train = X[:split,:]
    Y_train = Y[:split,:]
    training_batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=1024, shuffle=True)
    X_test = X[split:,:]
    Y_test = Y[split:,:]

    # Measure starting loss.
    net.eval()
    loss = get_loss(net(X_test), Y_test, debug_print=True)
    print(f'Starting test loss: {loss}.')
    # Train.
    optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9)
    # torch.autograd.set_detect_anomaly(True)
    min_test_loss = 999999
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
        with torch.no_grad():
            X_train_pred = net(X_train)
            X_test_pred = net(X_test)
            train_loss = get_loss(X_train_pred, Y_train, debug_print=False)
            test_loss = get_loss(X_test_pred, Y_test, debug_print=True)
        torch.cuda.empty_cache()
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(net, 'model_270K_v003.pt')
        print(f'Epoch finished in {elapsed_time:.1f}s. Train loss: {train_loss:.4f}. Test loss: {test_loss:.4f}.')

if __name__ == '__main__':
    test_null_training()