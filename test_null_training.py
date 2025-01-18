import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

class NullTrainingNetwork(nn.Module):
    def __init__(self):
        super(NullTrainingNetwork, self).__init__()
        # input vector: see PFState.to_tensor
        input_size = 160
        # output vector:
        #   [1]: value: 1 if the current player is winning, -1 if the current player is losing
        #   [26]: (policy) place a piece here
        #   [26*26]: (policy) move a piece from space to space
        #   [26*4]: (policy) from space A, push in direction B
        output_size = 807
        hidden_layers = [input_size, 208, 104, 104, 104, 208] # "270K"
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
    
class NullTrainingNetworkV3(nn.Module):
    def __init__(self):
        super(NullTrainingNetworkV3, self).__init__()
        input_size = 160
        hidden_layer_size = 192 # "520K"
        output_size = 807
        self.reshape_in = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size, bias=False),
            nn.BatchNorm1d(hidden_layer_size),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        residual_blocks = []
        for _ in range(3):
            residual_blocks.append(nn.Sequential(
                nn.Linear(hidden_layer_size, hidden_layer_size, bias=False),
                nn.BatchNorm1d(hidden_layer_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_layer_size, hidden_layer_size, bias=False),
                nn.BatchNorm1d(hidden_layer_size),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_layer_size, hidden_layer_size, bias=False),
                nn.BatchNorm1d(hidden_layer_size),
                nn.ReLU(),
                nn.Dropout(0.2),
            ))
        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.reshape_out = nn.Linear(hidden_layer_size, output_size)
        self.value_activation = nn.Tanh()
    
    def forward(self, x):
        features = self.reshape_in(x)
        for residual_block in self.residual_blocks:
            features = residual_block(features) + features
        features = self.reshape_out(features)
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
    # net = NullTrainingNetworkV3().cpu()
    net = torch.load('model_270K_v007.pt', weights_only=False).cpu()
    dropouts = 0
    for module in net.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.2
            dropouts += 1
    if dropouts > 0:
        print(f'Adjusted {dropouts} dropout layer{'' if dropouts == 1 else 's'}.')
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'Training network with {pytorch_total_params} total parameters.')
    print('Loading dataset...')
    dataset = torch.load('dataset_270K_v007_2470K.tnsr', weights_only=False, map_location='cpu')
    X = dataset['X']
    Y = dataset['Y']
    print(f'Loaded dataset. Inputs are of shape {X.shape}, outputs of shape {Y.shape}.')
    split = math.floor(len(X) * 0.9)
    X_train = X[:split,:].cuda()
    Y_train = Y[:split,:].cuda()
    training_batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=256, shuffle=True, drop_last=True)
    X_test = X[split:,:]
    Y_test = Y[split:,:]

    # Measure starting loss.
    net.eval()
    min_test_loss = get_loss(net(X_test), Y_test, test_loss=True)
    net.cuda()
    print(f'Starting test loss: {min_test_loss:.4f}.')
    # Train.
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # torch.autograd.set_detect_anomaly(True)
    torch.backends.cudnn.benchmark = True 
    for epoch in range(1000):
        net.train()
        start_time = time.time()
        total_train_loss = 0
        batches = 0
        for (batch_input, batch_output) in training_batches:
            predictions = net(batch_input)
            loss = get_loss(predictions, batch_output)
            total_train_loss += loss
            batches += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_time = time.time() - start_time
        net.eval()
        with torch.no_grad():
            net.cpu()
            X_test_pred = net(X_test)
            test_loss = get_loss(X_test_pred, Y_test, test_loss=True)
            net.cuda()
        # torch.cuda.empty_cache()
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(net, 'model_270K_v008rc2.pt')
        train_loss = total_train_loss / batches
        elapsed_time = time.time() - start_time
        print(f'Epoch {epoch + 1} finished in {elapsed_time:.1f}s ({batch_time:.1f}s batch time). Train/test loss: {train_loss:.4f}/{test_loss:.4f}.')

if __name__ == '__main__':
    test_null_training()