# fedpylot/federated/oort_integration/oort_selector.py

import numpy as np
import math
from random import Random
from collections import OrderedDict
import logging

class OortSelector:
    """Oort client selection - simplified for FedPylot"""
    
    def __init__(self, args, sample_seed=233):
        self.totalArms = OrderedDict()
        self.round_count = 0
        
        self.exploration = args.get('exploration_factor', 0.9)
        self.decay_factor = args.get('exploration_decay', 0.95)
        self.exploration_min = args.get('exploration_min', 0.2)
        
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        
        self.round_threshold = args.get('round_threshold', 10)
        self.round_penalty = args.get('round_penalty', 2.0)
        self.cut_off_util = args.get('cut_off_util', 0.7)
        
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        
    def register_client(self, clientId, feedbacks):
        """Register client with initial stats"""
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {
                'reward': feedbacks['reward'],
                'duration': feedbacks['duration'],
                'time_stamp': self.round_count,
                'count': 0,
                'status': True
            }
            self.unexplored.add(clientId)
    
    def update_client_util(self, clientId, feedbacks):
        """Update client utility after training"""
        self.totalArms[clientId]['reward'] = feedbacks['reward']
        self.totalArms[clientId]['duration'] = feedbacks['duration']
        self.totalArms[clientId]['time_stamp'] = feedbacks['time_stamp']
        self.totalArms[clientId]['count'] += 1
        self.unexplored.discard(clientId)
        self.successfulClients.add(clientId)
    
    def select_participants(self, num_clients, feasible_clients, cur_time):
        """Main selection logic"""
        self.round_count = cur_time
        
        # Compute scores for each client
        scores = {}
        for clientId in feasible_clients:
            if clientId not in self.totalArms:
                continue
                
            if self.totalArms[clientId]['count'] > 0:
                # UCB score: reward + exploration bonus
                reward = self.totalArms[clientId]['reward']
                staleness = cur_time - self.totalArms[clientId]['time_stamp']
                
                score = reward + math.sqrt(
                    0.1 * math.log(cur_time) / 
                    max(1, self.totalArms[clientId]['time_stamp'])
                )
                
                # Penalize slow clients
                duration = self.totalArms[clientId]['duration']
                if duration > self.round_threshold:
                    score *= (self.round_threshold / duration) ** self.round_penalty
                
                scores[clientId] = score
        
        # Select top-k by score
        clientLakes = list(scores.keys())
        if len(clientLakes) == 0:
            return list(feasible_clients)[:num_clients]
        
        exploitLen = int(num_clients * (1.0 - self.exploration))
        sortedClients = sorted(scores, key=scores.get, reverse=True)
        
        # Take top clients above cutoff
        cut_off_util = scores[sortedClients[min(exploitLen, len(sortedClients)-1)]] * self.cut_off_util
        pickedClients = [c for c in sortedClients if scores[c] >= cut_off_util]
        
        # Sample from top clients
        if len(pickedClients) > exploitLen:
            totalScore = sum(scores[c] for c in pickedClients)
            probs = [scores[c]/totalScore for c in pickedClients]
            pickedClients = list(np.random.choice(
                pickedClients, exploitLen, p=probs, replace=False
            ))
        
        # Add exploration
        unexplored = list(self.unexplored & set(feasible_clients))
        exploreLen = min(len(unexplored), num_clients - len(pickedClients))
        if exploreLen > 0:
            pickedClients += list(np.random.choice(
                unexplored, exploreLen, replace=False
            ))
        
        # Fill remaining
        while len(pickedClients) < num_clients:
            nextId = self.rng.choice(list(feasible_clients))
            if nextId not in pickedClients:
                pickedClients.append(nextId)
        
        self.exploration = max(
            self.exploration * self.decay_factor, 
            self.exploration_min
        )
        
        return pickedClients[:num_clients]