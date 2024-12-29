import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import AlphaPushNetwork
from pushfight import PFPiece, PFState
from training import eval_match, generate_training_data

model_prefix = 'model_750K'
models = glob.glob(f'{model_prefix}_v*.pt')
version = 1
if len(models) > 0:
    models.sort()
    net = torch.load(models[-1], weights_only=False).cpu()
    version = int(models[-1][-6:-3])
    print(f'Loaded model version {version} from {models[-1]}.')
else:
    net = AlphaPushNetwork().cpu()
    filename = f'{model_prefix}_v{version:03}.pt'
    torch.save(net, filename)
    print(f'Created fresh model at {filename}.')
pytorch_total_params = sum(p.numel() for p in net.parameters())
print(f'{pytorch_total_params} total parameters.')
loss_fn_value = nn.MSELoss()
loss_fn_policy = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=.1)

while True:
    # Self-play.
    batches_per_iteration = 20
    for batch in range(batches_per_iteration):
        print(f'Generating self-play batch {batch + 1} of {batches_per_iteration}...')
        training_inputs, training_outputs = generate_training_data(net)
        training_policies = training_outputs[:,1:]
        policy_masks = (training_policies == 0)
        print(f'Batch of size {training_inputs.shape[0]} generated. Updating network...')
        net.train()
        predictions = net(training_inputs)
        loss_value = loss_fn_value(predictions[:,:1], training_outputs[:,:1])
        predicted_policy = predictions[:,1:].masked_fill(policy_masks, float('-inf'))
        predicted_policy = nn.functional.softmax(predicted_policy, dim=1)
        prediction_zeroes = (predicted_policy == 0)
        assert torch.equal(prediction_zeroes, policy_masks)
        loss_policy = torch.mean(torch.sum(torch.abs(predicted_policy - training_policies), dim=1))
        loss = loss_value + loss_policy
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('Network updated.')

    # Evaluation.
    old_net = torch.load(f'{model_prefix}_v{version:03}.pt', weights_only=False).cpu()
    new_is_better = eval_match(old_net, net)
    if new_is_better:
        version += 1
        filename = f'{model_prefix}_v{version:03}.pt'
        print(f'Saving new network {filename}...')
        torch.save(net, filename)