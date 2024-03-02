#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 13:23:41 2023

@author: feijen
"""



import os
# import json
# import pprint as pp

import joblib
import torch
import torch.optim as optim
# from tensorboard_logger import Logger as TbLogger
from torch.nn import DataParallel

# from nets.critic_network import CriticNetwork
# from options import get_options
#from attention.train import get_inner_model
# from reinforce_baselines import NoBaseline, ExponentialBaseline, CriticBaseline, RolloutBaseline, WarmupBaseline
from attention.nets.attention_model import AttentionModel
from attention.nets.attention_model import set_decode_type
from torch.utils.data import DataLoader
# from nets.pointer_network import PointerNetwork, CriticNetworkLSTM
from attention.utils import torch_load_cpu, load_problem, move_to

from tqdm import tqdm

import numpy as np

# global use_cuda, device

global instanceName, testName, currentRunNumber

def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def init():

    global use_cuda, device, model, trainfreq, optimizer, baseline, newMemory
    print("Initialize attention model for neighborhood creation.")
    use_cuda = False
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    embedding_dim = 128
    hidden_dim = 128
    problem = load_problem("nc")
    n_encode_layers = 3
    normalization = 'batch'
    tanh_clipping = 10.0
    checkpoint_encoder = False
    shrink_size = None
    
    lr_model = 0.0001
    
    
    baseline = 10
    
    model = AttentionModel(
        embedding_dim,
        hidden_dim,
        problem,
        n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=normalization,
        tanh_clipping=tanh_clipping,
        checkpoint_encoder=checkpoint_encoder,
        shrink_size=shrink_size,
        num_route_features = 65
    ).to(device)
    
    newMemory = []
        
    trainfreq = 10
    
    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': lr_model}]
    ) #I deleted something here to do with the baseline and learnable parameters..?

def rollout(model, dataset, eval_batch_size, no_progress_bar):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat):
        with torch.no_grad():
            cost, _ = model(move_to(bat, device))
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat)
        for bat
        in tqdm(DataLoader(dataset, batch_size=eval_batch_size), disable=no_progress_bar)
    ], 0)

def sample_nhood_attention(features, nrRoutes):
    global device
    # create tensor of size (1,#vehicles,len(features)
    set_decode_type(model, "greedy")
    model.eval()

    # anchor = anchor_recommendations[0]
    # recs = anchor_recommendations[1:]
    features = torch.Tensor([features])
    _, _, pi = model(move_to(features,device),return_pi=True, nr_routes=nrRoutes)
    
    # print(pi)
    
    return pi[0]

def update_baseline(baseline, newMemory):
    alpha = 0.1
    
    average_improvement = np.mean([impr for _,_,impr in newMemory])
    
    return alpha * average_improvement + (1-alpha) * baseline


def compute_cost(improvement, baseline):
    if improvement <= baseline:
        return 0
    return - (improvement + 1) / (baseline + 1)

def train(features, pi, improvement,nrRoutes, continuousLearning, iteration):
    global newMemory, baseline, instanceName, testName, currentRunNumber
    
    # print(len(features))
    
    newMemory.append((features, pi, improvement))
    
    if len(newMemory) >= trainfreq:
        
        baseline = update_baseline(baseline, newMemory)
        
        # print('basleline updated to ', baseline)
        
        costs = torch.Tensor([compute_cost(improvement, baseline) for _,_,improvement in newMemory])
        # in case of experience replay somehow need to save these costs for later
        
        if continuousLearning == 'completerun':
            sampleDir = "../temp - sample dump/"
            sample_name = f"contlearn_samples_{instanceName}_{testName}_{currentRunNumber}_{iteration}.smpl"
            joblib.dump((newMemory, costs), sampleDir + sample_name )

        ExperienceReplay = False
        if ExperienceReplay:
            pass
        else:
            trainMemory = newMemory
            trainCosts = costs

        #The NC problems can have different number of routes (since there can be a variable number in the last shell)
        #Therefore, the features in trainMemory can have different length
        #Therefore, we cannot make 1 tensor out of them, and we do the training one by one:
            
        for sample, trainCost in zip(trainMemory, trainCosts):
            features,selected,_ = sample
            
            update_with_sample(features, selected, nrRoutes, trainCost)
            
            newMemory = []
            
        
def update_with_sample(features, selected, nrRoutes, costs):
    selected = torch.tensor(selected)
    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")
    
    x = torch.Tensor([features])
    x = move_to(x, device)
    
    # print('trainmemory', trainMemory)
    
    # print('[selected for _,selected,_ in trainMemory]', [selected for _,selected,_ in trainMemory])
    # print('torch.stack([selected for _,selected,_ in trainMemory])', torch.stack([selected for _,selected,_ in trainMemory]))
    
    # print(selected.size(), selected)
    
    # selected = torch.Tensor([selected])
    
    # selected = torch.stack([selected for _,selected,_ in trainMemory])

    # Evaluate model, get costs and log probabilities
    _, log_likelihood = model(x, nr_routes=nrRoutes, pi = selected[None])

    # print(costs)

    # Calculate loss
    loss = ((costs) * log_likelihood).mean()
    
    optimizer.zero_grad()
    loss.backward()

    # Perform backward pass and optimization step
    # Clip gradient norms and get (clipped) gradient norms for logging
    # grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()
    
    
def load_model(runNumber, saveDir, testName):
    
    load_path = os.path.join(saveDir, f'model_{testName}_forRun_{runNumber}.pt')
    
    print('  [*] Loading data from {}'.format(load_path))
    load_data = torch_load_cpu(load_path)
    
    # Overwrite model parameters by parameters to load
    model_ = get_inner_model(model)
    model_.load_state_dict({**model_.state_dict(), **load_data.get('model', {})})
    
    # Load optimizer state
    if 'optimizer' in load_data:
        optimizer.load_state_dict(load_data['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(device)
                    
def save_model(nextRunNumber, saveDir, testName):
    
    print('Saving model and state...')
    torch.save(
        {
            'model': get_inner_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all(),
            # 'baseline': baseline.state_dict()
        },
        os.path.join(saveDir, f'model_{testName}_forRun_{nextRunNumber}.pt')
    )