import math
import random
import time
import torch
import torch.nn as nn

from pushfight import PFPiece, PFState

def build_win_in_one_dataset():
    inputs = []
    outputs = []
    DATASET_SIZE = 1000000
    BATCH_SIZE = 100000
    wins = 0
    print(f'Generating win-in-one dataset of size {DATASET_SIZE}.')
    while len(inputs) < DATASET_SIZE:
        state = PFState()
        for i in range(10):
            state.move(random.choice(state.moves))
        old_len = len(inputs)
        # Check if there's a mate in one.
        while state.winner == PFPiece.Empty:
            inputs.append(state.to_tensor())
            player = PFPiece.White if state.white_to_move else PFPiece.Black
            if any(state.get_move_winner(move) == player for move in state.moves):
                wins += 1
                outputs.append(1)
            else:
                outputs.append(0)
            state.move(random.choice(state.moves))
        if len(inputs) % BATCH_SIZE < old_len % BATCH_SIZE:
            print(f'Saving with {len(inputs)} examples ({(wins * 100 / len(inputs)):.2f}% wins).')
            torch.save({'X': torch.stack(inputs), 'Y': torch.tensor(outputs).unsqueeze(1).float()}, 'winset.pt')
    print('Done.')

class ValueOnlyNetwork(nn.Module):
    def __init__(self):
        super(ValueOnlyNetwork, self).__init__()
        output_size = 1
        input_size = 160
        hidden_layers = [input_size, 104, 104, 104, 104, 104, 26]
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
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def test_value_only_network():
    net = ValueOnlyNetwork().cuda()
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(f'Created network with {pytorch_total_params} total parameters.')
    print('Loading dataset...')
    winset = torch.load('winset.pt', weights_only=False)
    X = winset['X'].cuda()
    Y = winset['Y'].cuda()
    print(f'Loaded dataset. Inputs are of shape {X.shape}, outputs of shape {Y.shape}.')
    split = math.floor(len(X) * 0.9)
    X_train = X[:split,:]
    Y_train = Y[:split,:]
    training_batches = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=1024, shuffle=True)
    X_test = X[split:,:]
    Y_test = Y[split:,:]
    loss_fn = nn.MSELoss()
    # Measure starting loss.
    net.eval()
    loss = loss_fn(net(X_test), Y_test)
    print(f'Starting test loss: {loss}.')
    # Train.
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: .98 ** epoch)
    for epoch in range(100):
        print(f'Starting epoch {epoch + 1} of 100.')
        net.train()
        start_time = time.time()
        for (batch_input, batch_output) in training_batches:
            predictions = net(batch_input)
            loss = loss_fn(predictions, batch_output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        elapsed_time = time.time() - start_time
        net.eval()
        loss = loss_fn(net(X_test), Y_test)
        print(f'Finished in {elapsed_time:.1f}s. Test loss: {loss}.')

# build_win_in_one_dataset()
test_value_only_network()