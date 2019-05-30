#!/usr/bin/env python

"""
    train.py
"""

from __future__ import division
from __future__ import print_function

import sys
import argparse
import ujson as json
import numpy as np
from time import time

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from models import HINGCN_GS
from problem import NodeProblem
from helpers import set_seeds, to_numpy
from nn_modules import aggregator_lookup, prep_lookup, sampler_lookup
from lr import LRSchedule

# --
# Helpers

def evaluate(model, problem, batch_size, mode='val'):
    assert mode in ['test', 'val']
    preds, acts = [], []
    for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False, batch_size=batch_size):
        # print(ids.shape,targets.shape)
        preds.append(to_numpy(model(ids, problem.feats, problem.adj, problem.edge_emb, train=False)))
        acts.append(to_numpy(targets))
    
    return problem.metric_fn(np.vstack(acts), np.vstack(preds))

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--problem-path', type=str, required=True)
    parser.add_argument('--no-cuda', action="store_true")
    
    # Optimization params
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr-init', type=float, default=0.1)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    
    # Architecture params
    parser.add_argument('--sampler-class', type=str, default='sparse_uniform_neighbor_sampler')
    parser.add_argument('--aggregator-class', type=str, default='edge_emb_attn')
    parser.add_argument('--prep-class', type=str, default='identity')
    parser.add_argument('--mpaggr-class', type=str, default='metapath')
    parser.add_argument('--edgeupt-class', type=str, default='edge')
    parser.add_argument('--concat-node', type=bool, default=False)
    parser.add_argument('--concat-edge', type=bool, default=False)

    parser.add_argument('--n-train-samples', type=str, default='8,8')
    parser.add_argument('--n-val-samples', type=str, default='8,8')
    parser.add_argument('--output-dims', type=str, default='16,16')
    
    # Logging
    parser.add_argument('--log-interval', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--show-test', action="store_true")
    
    # --
    # Validate args
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert args.prep_class in prep_lookup.keys(), 'parse_args: prep_class not in %s' % str(prep_lookup.keys())
    assert args.aggregator_class in aggregator_lookup.keys(), 'parse_args: aggregator_class not in %s' % str(aggregator_lookup.keys())
    assert args.batch_size > 1, 'parse_args: batch_size must be > 1'
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # Load problem
    schemes = ['APA','APAPA','APCPA']#
    problem = NodeProblem(problem_path=args.problem_path, cuda=args.cuda, schemes=schemes)
    
    # --
    # Define model
    
    n_train_samples =  list(map(int,args.n_train_samples.split(',')))
    n_val_samples = list(map(int,args.n_val_samples.split(',')))
    output_dims = list(map(int,args.output_dims.split(',') ))
    model = HINGCN_GS(**{
        "sampler_class" : sampler_lookup[args.sampler_class],
        
        "prep_class" : prep_lookup[args.prep_class],
        "aggregator_class" : aggregator_lookup[args.aggregator_class],
        "mpaggr_class": aggregator_lookup[args.mpaggr_class],
        "edgeupt_class": aggregator_lookup[args.edgeupt_class],

        "input_dim" : problem.feats_dim,
        "edge_dim"  : problem.edge_dim,
        "schemes"   : problem.schemes,
        "n_nodes"   : problem.n_nodes,
        "n_classes" : problem.n_classes,
        "layer_specs" : [
            {
                "n_train_samples" : n_train_samples[0],
                "n_val_samples" : n_val_samples[0],
                "output_dim" : output_dims[0],
                "activation" : F.relu,
                "concat_node" : args.concat_node,
                "concat_edge" : args.concat_edge,
            },
            {
                "n_train_samples" : n_train_samples[1],
                "n_val_samples" : n_val_samples[1],
                "output_dim" : output_dims[1],
                "activation" : F.relu,  # lambda x: x
                "concat_node": args.concat_node,
                "concat_edge": args.concat_edge,
            },
        ],
        
        "lr_init" : args.lr_init,
        "lr_schedule" : args.lr_schedule,
        "weight_decay" : args.weight_decay,
    })
    
    if args.cuda:
        model = model.cuda()
    
    print(model, file=sys.stderr)
    
    # --
    # Train
    
    set_seeds(args.seed)
    
    start_time = time()
    val_metric = None
    for epoch in range(args.epochs):
        train_loss=0
        # Train
        _ = model.train()
        for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size):
            model.set_progress((epoch + epoch_progress) / args.epochs)
            loss, preds = model.train_step(
                ids=ids, 
                feats=problem.feats,
                edge_emb=problem.edge_emb,
                adjs=problem.adj,
                targets=targets,
                loss_fn=problem.loss_fn,
            )
            train_loss+=loss.item()
            train_metric = problem.metric_fn(to_numpy(targets), to_numpy(preds))
            print(json.dumps({
                "epoch" : epoch,
                "epoch_progress" : epoch_progress,
                "train_metric" : train_metric,
                "time" : time() - start_time,
            }, double_precision=5))
            sys.stdout.flush()


        # Evaluate
        if epoch%args.log_interval==0:
            _ = model.eval()
            val_metric = evaluate(model, problem, batch_size=args.batch_size, mode='val')

            print(json.dumps({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metric": val_metric,
            }, double_precision=5))
            sys.stdout.flush()

    print('-- done --', file=sys.stderr)
    print(json.dumps({
        "epoch" : epoch,
        "train_metric" : train_metric,
        "val_metric" : val_metric,
        "time" : time() - start_time,
    }, double_precision=5))
    sys.stdout.flush()
    
    if args.show_test:
        print(json.dumps({
            "test_f1" : evaluate(model, problem, batch_size=args.batch_size, mode='test')
        }, double_precision=5))

