import glob
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from network import AlphaPushNetwork, NullNet
from training import eval_match, generate_training_data

model_prefix = 'model_320K'
models = glob.glob(f'{model_prefix}_v*.pt')
version = 1
if len(models) > 0:
    models.sort()
    net = torch.load(f'{model_prefix}_checkpoint.pt', weights_only=False).cpu() # torch.load(models[-1], weights_only=False).cpu()
    version = int(models[-1][-6:-3])
    print(f'Loaded model version {version} from {models[-1]}.')
else:
    net = AlphaPushNetwork().cpu()
    filename = f'{model_prefix}_v{version:03}.pt'
    torch.save(net, filename)
    print(f'Created fresh model at {filename}.')
pytorch_total_params = sum(p.numel() for p in net.parameters())
print(f'{pytorch_total_params} total parameters.')
training_net = NullNet()
loss_fn_value = nn.MSELoss()
loss_policy_weight = 0.5
optimizer = torch.optim.SGD(net.parameters(), lr=0.2, momentum=0.9)

while True:
    # Self-play.
    batch_size = 128
    examples_per_iteration = batch_size * 200
    generation_parallelism = 20
    generation_passes = 4
    for generation_pass in range(generation_passes):
        examples_this_pass = examples_per_iteration / generation_passes
        print(f'Starting generation pass {generation_pass + 1} of {generation_passes}.')
        start_time = time.time()
        training_inputs, training_outputs = generate_training_data(training_net, min_training_items=examples_this_pass, parallelism=generation_parallelism)
        print(f'Generated {len(training_inputs)} training examples in {math.floor(time.time() - start_time)}s.')
        num_batches = len(training_inputs) // batch_size
        total_loss = 0
        total_value_loss = 0
        total_policy_loss = 0
        for i in range(0, num_batches * batch_size, batch_size):
            batch_inputs = training_inputs[i:i+batch_size]
            batch_outputs = training_outputs[i:i+batch_size]
            training_policies = batch_outputs[:,1:]
            policy_masks = (training_policies == 0)
            net.train()
            predictions = net(batch_inputs)
            loss_value = loss_fn_value(predictions[:,:1], batch_outputs[:,:1])
            predicted_policy = predictions[:,1:].masked_fill(policy_masks, -1e9)
            loss_policy = F.cross_entropy(predicted_policy, training_policies)
            loss = loss_value + loss_policy * loss_policy_weight
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss
            total_value_loss += loss_value
            total_policy_loss += loss_policy
        print(f'Average loss: {(total_loss / num_batches):.3f}, value: {(total_value_loss / num_batches):.3f}, policy: {(total_policy_loss / num_batches):.3f}')
        torch.save(net, f'{model_prefix}_checkpoint.pt')

    # Evaluation.
    old_net = torch.load(f'{model_prefix}_v{version:03}.pt', weights_only=False).cpu()
    print('Running eval match...')
    new_is_better = eval_match(old_net, net)
    if new_is_better:
        version += 1
        filename = f'{model_prefix}_v{version:03}.pt'
        print(f'Saving new network {filename}...')
        torch.save(net, filename)