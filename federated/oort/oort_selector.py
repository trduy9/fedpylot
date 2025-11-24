import numpy as np
import math
from random import Random
from collections import OrderedDict
import logging

class OortSelector:
    """
    Oort client selection - Full implementation matching original Oort
    Implements UCB-based selection with system utility, statistical utility,
    pacer mechanism, and blacklist
    """
    
    def __init__(self, args, sample_seed=233):
        # Core data structures
        self.totalArms = OrderedDict()
        self.training_round = 0
        
        # Exploration parameters
        self.exploration = args.get('exploration_factor', 0.9)
        self.decay_factor = args.get('exploration_decay', 0.95)
        self.exploration_min = args.get('exploration_min', 0.2)
        self.alpha = args.get('exploration_alpha', 0.3)  # For staleness-based exploration
        
        # Random number generator
        self.rng = Random()
        self.rng.seed(sample_seed)
        self.unexplored = set()
        
        # System utility parameters
        self.round_threshold = args.get('round_threshold', 10)
        self.round_penalty = args.get('round_penalty', 2.0)
        self.round_prefer_duration = float('inf')
        
        # Pacer mechanism parameters
        self.pacer_step = args.get('pacer_step', 20)
        self.pacer_delta = args.get('pacer_delta', 5)
        self.last_util_record = 0
        
        # Selection parameters
        self.cut_off_util = args.get('cut_off_util', 0.7)
        self.sample_window = args.get('sample_window', 5.0)
        self.clip_bound = args.get('clip_bound', 0.95)
        
        # Blacklist parameters
        self.blacklist_rounds = args.get('blacklist_rounds', -1)
        self.blacklist_max_len = args.get('blacklist_max_len', 0.3)
        self.blacklist = set()
        
        # Tracking
        self.exploitUtilHistory = []
        self.exploreUtilHistory = []
        self.exploitClients = []
        self.exploreClients = []
        self.successfulClients = set()
        
        # Store args for later use
        self.args = args
        
        np.random.seed(sample_seed)
    
    def register_client(self, clientId, feedbacks):
        """
        Register client with initial statistics
        
        Args:
            clientId: Unique client identifier
            feedbacks: Dict containing 'reward' and 'duration'
        """
        if clientId not in self.totalArms:
            self.totalArms[clientId] = {
                'reward': feedbacks['reward'],
                'duration': feedbacks['duration'],
                'time_stamp': self.training_round,
                'count': 0,
                'status': True
            }
            self.unexplored.add(clientId)
    
    def update_client_util(self, clientId, feedbacks):
        """
        Update client utility after training round
        
        Args:
            clientId: Client identifier
            feedbacks: Dict with 'reward', 'duration', 'time_stamp', 'status'
        """
        if clientId not in self.totalArms:
            logging.warning(f"Client {clientId} not registered before update")
            return
            
        self.totalArms[clientId]['reward'] = feedbacks['reward']
        self.totalArms[clientId]['duration'] = feedbacks['duration']
        self.totalArms[clientId]['time_stamp'] = feedbacks['time_stamp']
        self.totalArms[clientId]['count'] += 1
        self.totalArms[clientId]['status'] = feedbacks.get('status', True)
        
        self.unexplored.discard(clientId)
        self.successfulClients.add(clientId)
    
    def update_duration(self, clientId, duration):
        """Update client duration separately"""
        if clientId in self.totalArms:
            self.totalArms[clientId]['duration'] = duration
    
    def pacer(self):
        """
        Pacer mechanism to adaptively adjust round_threshold
        Monitors exploitation utility and adjusts threshold to balance
        statistical and system utility
        """
        # Calculate average utility for last round
        lastExplorationUtil = self.calculateSumUtil(self.exploreClients)
        lastExploitationUtil = self.calculateSumUtil(self.exploitClients)
        
        self.exploreUtilHistory.append(lastExplorationUtil)
        self.exploitUtilHistory.append(lastExploitationUtil)
        
        # Reset successful clients for next round
        self.successfulClients = set()
        
        # Check if we should adjust pacer
        if self.training_round >= 2 * self.pacer_step and self.training_round % self.pacer_step == 0:
            utilLastPacerRounds = sum(self.exploitUtilHistory[-2*self.pacer_step:-self.pacer_step])
            utilCurrentPacerRounds = sum(self.exploitUtilHistory[-self.pacer_step:])
            
            # Utility becomes flat -> relax pacer (allow slower clients)
            if abs(utilCurrentPacerRounds - utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                self.round_threshold = min(100., self.round_threshold + self.pacer_delta)
                self.last_util_record = self.training_round - self.pacer_step
                logging.debug(f"Pacer increased at round {self.training_round} to {self.round_threshold}")
            
            # Utility changes sharply -> tighten pacer (prefer faster clients)
            elif abs(utilCurrentPacerRounds - utilLastPacerRounds) >= utilLastPacerRounds * 5:
                self.round_threshold = max(self.pacer_delta, self.round_threshold - self.pacer_delta)
                self.last_util_record = self.training_round - self.pacer_step
                logging.debug(f"Pacer decreased at round {self.training_round} to {self.round_threshold}")
            
            logging.debug(f"Pacer check: utilLast={utilLastPacerRounds}, utilCurrent={utilCurrentPacerRounds}")
        
        logging.info(f"Round {self.training_round}: exploitUtil={lastExploitationUtil:.4f}, "
                    f"exploreUtil={lastExplorationUtil:.4f}, pacer={self.round_threshold}")
    
    def calculateSumUtil(self, clientList):
        """Calculate average utility for a list of clients"""
        cnt, cntUtil = 1e-4, 0
        
        for client in clientList:
            if client in self.successfulClients:
                cnt += 1
                cntUtil += self.totalArms[client]['reward']
        
        return cntUtil / cnt
    
    def get_blacklist(self):
        """
        Generate blacklist of clients selected too frequently
        Prevents over-selection of certain clients
        """
        blacklist = []
        
        if self.blacklist_rounds != -1:
            # Sort clients by selection count (descending)
            sorted_client_ids = sorted(
                list(self.totalArms.keys()), 
                reverse=True,
                key=lambda k: self.totalArms[k]['count']
            )
            
            # Add clients exceeding threshold to blacklist
            for clientId in sorted_client_ids:
                if self.totalArms[clientId]['count'] > self.blacklist_rounds:
                    blacklist.append(clientId)
                else:
                    break
            
            # Cap blacklist size
            predefined_max_len = int(self.blacklist_max_len * len(self.totalArms))
            if len(blacklist) > predefined_max_len:
                logging.warning(f"Blacklist exceeds threshold, capping at {predefined_max_len}")
                blacklist = blacklist[:predefined_max_len]
        
        return set(blacklist)
    
    def get_norm(self, aList, clip_bound=0.95, thres=1e-4):
        """
        Get normalization parameters with optional clipping
        
        Returns:
            max, min, range, avg, clip_value
        """
        if len(aList) == 0:
            return 0., 0., thres, 0., 0.
        
        aList_sorted = sorted(aList)
        clip_value = aList_sorted[min(int(len(aList) * clip_bound), len(aList) - 1)]
        
        _max = max(aList)
        _min = min(aList) * 0.999
        _range = max(_max - _min, thres)
        _avg = sum(aList) / max(1e-4, float(len(aList)))
        
        return float(_max), float(_min), float(_range), float(_avg), float(clip_value)
    
    def select_participants(self, num_clients, feasible_clients, cur_time):
        """
        Main client selection algorithm using UCB with system/statistical utility
        
        Args:
            num_clients: Number of clients to select
            feasible_clients: List of available client IDs
            cur_time: Current training round
            
        Returns:
            List of selected client IDs
        """
        self.training_round = cur_time
        self.blacklist = self.get_blacklist()
        
        # Run pacer to adjust round_threshold
        self.pacer()
        
        # Filter out blacklisted and unavailable clients
        feasible_clients_set = set(feasible_clients)
        orderedKeys = [
            x for x in self.totalArms.keys() 
            if int(x) in feasible_clients_set and int(x) not in self.blacklist
        ]
        
        if len(orderedKeys) == 0:
            logging.warning("No feasible clients available after filtering")
            return list(feasible_clients)[:num_clients]
        
        # Update round_prefer_duration based on round_threshold
        if self.round_threshold < 100.:
            client_list = list(self.totalArms.keys())
            sortedDuration = sorted([self.totalArms[key]['duration'] for key in client_list])
            threshold_idx = min(
                int(len(sortedDuration) * self.round_threshold / 100.),
                len(sortedDuration) - 1
            )
            self.round_prefer_duration = sortedDuration[threshold_idx]
        else:
            self.round_prefer_duration = float('inf')
        
        # Collect rewards and staleness for normalization
        moving_reward, staleness = [], []
        for clientId in orderedKeys:
            if self.totalArms[clientId]['reward'] > 0:
                moving_reward.append(self.totalArms[clientId]['reward'])
                staleness.append(cur_time - self.totalArms[clientId]['time_stamp'])
        
        # Normalize rewards and staleness
        max_reward, min_reward, range_reward, avg_reward, clip_value = \
            self.get_norm(moving_reward, self.clip_bound)
        max_staleness, min_staleness, range_staleness, avg_staleness, _ = \
            self.get_norm(staleness, thres=1)
        
        # Calculate scores for exploitation
        scores = {}
        numOfExploited = 0
        
        for key in orderedKeys:
            if self.totalArms[key]['count'] > 0:
                # Clip reward to avoid outliers
                creward = min(self.totalArms[key]['reward'], clip_value)
                numOfExploited += 1
                
                # UCB score: normalized reward + exploration bonus (temporal uncertainty)
                normalized_reward = (creward - min_reward) / float(range_reward)
                exploration_bonus = math.sqrt(
                    0.1 * math.log(cur_time) / self.totalArms[key]['time_stamp']
                )
                sc = normalized_reward + exploration_bonus
                
                # Apply system utility penalty for slow clients
                clientDuration = self.totalArms[key]['duration']
                if clientDuration > self.round_prefer_duration:
                    penalty = (float(self.round_prefer_duration) / max(1e-4, clientDuration)) ** self.round_penalty
                    sc *= penalty
                
                scores[key] = sc
        
        # === EXPLOITATION PHASE ===
        clientLakes = list(scores.keys())
        self.exploration = max(self.exploration * self.decay_factor, self.exploration_min)
        exploitLen = min(int(num_clients * (1.0 - self.exploration)), len(clientLakes))
        
        if exploitLen > 0:
            # Sort by score and apply cutoff
            sortedClientUtil = sorted(scores, key=scores.get, reverse=True)
            cut_off_util = scores[sortedClientUtil[min(exploitLen, len(sortedClientUtil)-1)]] * self.cut_off_util
            
            # Select clients above cutoff
            pickedClients = [
                clientId for clientId in sortedClientUtil 
                if scores[clientId] >= cut_off_util
            ]
            
            # Sample from top clients using score-based probability
            if len(pickedClients) > exploitLen:
                totalSc = max(1e-4, float(sum([scores[key] for key in pickedClients])))
                probs = [scores[key] / totalSc for key in pickedClients]
                pickedClients = list(np.random.choice(
                    pickedClients, exploitLen, p=probs, replace=False
                ))
            
            self.exploitClients = pickedClients
        else:
            pickedClients = []
            self.exploitClients = []
        
        # === EXPLORATION PHASE ===
        _unexplored = [x for x in list(self.unexplored) if int(x) in feasible_clients_set]
        exploreLen = min(len(_unexplored), num_clients - len(pickedClients))
        
        if exploreLen > 0:
            # Prioritize unexplored clients by initial reward (e.g., data size)
            init_reward = {}
            for cl in _unexplored:
                init_reward[cl] = self.totalArms[cl]['reward']
                clientDuration = self.totalArms[cl]['duration']
                
                # Apply system utility penalty
                if clientDuration > self.round_prefer_duration:
                    penalty = (float(self.round_prefer_duration) / max(1e-4, clientDuration)) ** self.round_penalty
                    init_reward[cl] *= penalty
            
            # Select top clients within sample window
            pickedUnexploredClients = sorted(
                init_reward, key=init_reward.get, reverse=True
            )[:min(int(self.sample_window * exploreLen), len(init_reward))]
            
            # Sample based on reward probability
            unexploredSc = float(sum([init_reward[key] for key in pickedUnexploredClients]))
            if unexploredSc > 0:
                probs = [init_reward[key] / max(1e-4, unexploredSc) for key in pickedUnexploredClients]
                pickedUnexplored = list(np.random.choice(
                    pickedUnexploredClients, exploreLen, p=probs, replace=False
                ))
            else:
                pickedUnexplored = pickedUnexploredClients[:exploreLen]
            
            self.exploreClients = pickedUnexplored
            pickedClients = pickedClients + pickedUnexplored
        else:
            self.exploreClients = []
            # No more unexplored clients
            if len(self.unexplored) == 0:
                self.exploration_min = 0.
                self.exploration = 0.
        
        # Fill remaining slots randomly if needed
        while len(pickedClients) < num_clients and len(orderedKeys) > 0:
            nextId = self.rng.choice(orderedKeys)
            if nextId not in pickedClients:
                pickedClients.append(nextId)
        
        # Logging
        top_k_score = []
        for i in range(min(3, len(pickedClients))):
            if i < len(self.exploitClients):
                clientId = self.exploitClients[i]
                if clientId in scores:
                    _score = (self.totalArms[clientId]['reward'] - min_reward) / range_reward
                    _staleness_norm = (cur_time - self.totalArms[clientId]['time_stamp'] - min_staleness) / float(range_staleness)
                    top_k_score.append((self.totalArms[clientId], [_score, _staleness_norm]))
        
        logging.info(f"Round {cur_time}: exploited={numOfExploited}, "
                    f"exploreLen={exploreLen}, unexplored={len(self.unexplored)}, "
                    f"exploration={self.exploration:.3f}, round_threshold={self.round_threshold}, "
                    f"top_scores={top_k_score}")
        
        return pickedClients[:num_clients]
    
    def get_median_reward(self):
        """Get mean reward of feasible (non-blacklisted) clients"""
        feasible_rewards = [
            self.totalArms[x]['reward'] 
            for x in list(self.totalArms.keys()) 
            if int(x) not in self.blacklist
        ]
        
        if len(feasible_rewards) > 0:
            return sum(feasible_rewards) / float(len(feasible_rewards))
        
        return 0
    
    def get_client_reward(self, clientId):
        """Get client statistics"""
        if clientId in self.totalArms:
            return self.totalArms[clientId]
        return None
    
    def getAllMetrics(self):
        """Get all client metrics"""
        return self.totalArms