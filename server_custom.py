import pickle
import numpy as np
import random
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import math
from scipy.stats import beta

class ClientSelection:
    
    def __init__(self):
        self.client_selections = {}  # Track number of selections per client
        self.client_rewards = {}     # Track rewards (global accuracy gain) per client
        self.client_beta_params = {} # Track Beta distribution parameters for Thompson Sampling

    def client_selection_fedbandits(self, clients, args: dict) -> list:
        round_num = args["round"]
        num_clients_per_round = args["num_clients_per_round"]
        bandit_algo = args.get("bandit_algo", "ucb")  # ucb, ts, or exp3
        epsilon = args.get("epsilon", 0.1)  # ε-greedy probability
        initial_random_rounds = args.get("initial_random_rounds", 5)  # Number of initial random rounds
        exploration_param = args.get("exploration_param", math.sqrt(2))  # UCB exploration parameter

        # Initialize tracking for each client
        if round_num == 1:
            self.client_selections = {client.cid: 0 for client in clients}
            self.client_rewards = {client.cid: [] for client in clients}
            self.client_beta_params = {client.cid: {'alpha': 1, 'beta': 1} for client in clients}  # Beta(1,1) for TS
            self.client_weights = {client.cid: 1.0 for client in clients}  # EXP3 weights

        # ε-greedy: Random selection with probability ε or in initial random rounds
        if round_num <= initial_random_rounds or random.random() < epsilon:
            selected_cids = np.random.choice([client.cid for client in clients], num_clients_per_round, replace=False).tolist()
            for cid in selected_cids:
                self.client_selections[cid] += 1
            return selected_cids

        # Bandit algorithm selection
        if bandit_algo == "ucb":
            # UCB: Select clients with highest UCB scores
            total_selections = sum(self.client_selections.values()) + 1e-10
            ucb_scores = {}
            for client in clients:
                cid = client.cid
                selections = self.client_selections[cid] + 1e-10
                avg_reward = sum(self.client_rewards[cid]) / len(self.client_rewards[cid]) if self.client_rewards[cid] else 0
                ucb_scores[cid] = avg_reward + exploration_param * math.sqrt(math.log(total_selections) / selections)
            selected_cids = sorted(ucb_scores, key=ucb_scores.get, reverse=True)[:num_clients_per_round]

        elif bandit_algo == "ts":
            # Thompson Sampling: Sample from Beta distribution
            samples = {}
            for client in clients:
                cid = client.cid
                alpha = self.client_beta_params[cid]['alpha']
                beta_param = self.client_beta_params[cid]['beta']
                samples[cid] = beta.rvs(alpha, beta_param)
            selected_cids = sorted(samples, key=samples.get, reverse=True)[:num_clients_per_round]

        elif bandit_algo == "exp3":
            # EXP3: Sample clients based on weights
            total_weight = sum(self.client_weights.values()) + 1e-10
            probabilities = {cid: self.client_weights[cid] / total_weight for cid in self.client_weights}
            selected_cids = np.random.choice(
                [client.cid for client in clients],
                size=num_clients_per_round,
                replace=False,
                p=[probabilities[cid] for cid in [client.cid for client in clients]]
            ).tolist()

        else:
            raise ValueError(f"Unknown bandit algorithm: {bandit_algo}")

        # Update selection counts
        for cid in selected_cids:
            self.client_selections[cid] += 1

        return selected_cids

    def update_rewards(self, selected_cids, client_list, round_id, server, prev_accuracy):
        """Update rewards based on global accuracy gain."""
        current_accuracy = server.test_metrics.get(round_id, {}).get('accuracy', 0)
        reward = current_accuracy - prev_accuracy if prev_accuracy is not None else current_accuracy
        reward = max(0, min(reward, 1))  # Normalize to [0,1] for TS and EXP3
        for cid in selected_cids:
            self.client_rewards[cid].append(reward)
            # Update Beta parameters for Thompson Sampling
            self.client_beta_params[cid]['alpha'] += reward
            self.client_beta_params[cid]['beta'] += (1 - reward)
            # Update weights for EXP3
            gamma = 0.1  # EXP3 exploration parameter
            total_weight = sum(self.client_weights.values()) + 1e-10
            prob = self.client_weights[cid] / total_weight
            self.client_weights[cid] *= math.exp((gamma * reward / prob) / len(client_list))

class Aggregation:
    def __init__(self):
        pass
    
    def aggregate_fedavg(self, round, selected_cids, client_list, update_client_models=True):
        global_model = OrderedDict()
        client_local_weights = client_list[0].model.to("cpu").state_dict()
        
        for layer in client_local_weights:
            shape = client_local_weights[layer].shape
            global_model[layer] = torch.zeros(shape)

        client_weights = list()
        n_k = list()
        for client_id in selected_cids:
            client_weights.append(client_list[client_id].model.to("cpu").state_dict())
            n_k.append(client_list[client_id].num_items)

        n_k = np.array(n_k)
        n_k = n_k / sum(n_k)
        
        for i, weights in enumerate(client_weights):
            for layer in weights.keys():
                global_model[layer] += (weights[layer] * n_k[i])

        if update_client_models:
            for client in client_list:
                client.model.load_state_dict(global_model)

        return global_model, client_list
    
class Server(ClientSelection, Aggregation):
    def __init__(self, logger, device, model_class, model_args, data_path, dataset_id, test_batch_size):
        ClientSelection.__init__(self)
        Aggregation.__init__(self)
        
        self.id = "server"
        self.device = device
        self.logger = logger
        self.model = model_class(self.id, model_args)
        
        _, self.test_data = self.model.load_data(logger, data_path, dataset_id, self.id, None, test_batch_size)

        self.test_metrics = dict()

    def test(self, round_id):
        data = self.test_data
        self.test_metrics[round_id] = self.model.test_model(self.logger, data)