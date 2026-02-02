from .oort_selector import OortSelector
import logging
import math
import os
import csv
class OortClientSampler:
    """
    Wrapper for Oort client selection - Full implementation
    Manages client registration, updates, and selection for federated learning
    """
    
    def __init__(self, args):
        """
        Initialize Oort client sampler
        
        Args:
            args: Configuration dict or argparse.Namespace with:
                - exploration_factor: Initial exploration rate (default: 0.9)
                - exploration_decay: Exploration decay rate (default: 0.95)
                - exploration_min: Minimum exploration rate (default: 0.2)
                - exploration_alpha: Staleness weight (default: 0.3)
                - round_threshold: System utility threshold % (default: 10)
                - round_penalty: Penalty for slow clients (default: 2.0)
                - cut_off_util: Score cutoff multiplier (default: 0.7)
                - sample_window: Exploration window multiplier (default: 5.0)
                - clip_bound: Reward clipping percentile (default: 0.95)
                - pacer_step: Rounds between pacer adjustments (default: 20)
                - pacer_delta: Pacer adjustment step (default: 5)
                - blacklist_rounds: Max rounds before blacklist (default: -1, disabled)
                - blacklist_max_len: Max blacklist fraction (default: 0.3)
        """
        self.args = args if isinstance(args, dict) else vars(args)
        
        # Initialize Oort selector with full configuration
        selector_config = {
            'exploration_factor': self.args.get('exploration_factor', 1.0),
            'exploration_decay': self.args.get('exploration_decay', 0.95),
            'exploration_min': self.args.get('exploration_min', 0.45),
            'exploration_alpha': self.args.get('exploration_alpha', 0.3),
            'round_threshold': self.args.get('round_threshold', 60),
            'round_penalty': self.args.get('round_penalty', 0.0),
            'cut_off_util': self.args.get('cut_off_util', 0.7),
            'sample_window': self.args.get('sample_window', 5.0),
            'clip_bound': self.args.get('clip_bound', 0.95),
            'pacer_step': self.args.get('pacer_step', 5),
            'pacer_delta': self.args.get('pacer_delta', 10),
            'blacklist_rounds': self.args.get('blacklist_rounds', 12),
            'blacklist_max_len': self.args.get('blacklist_max_len', 0.3),
            'utility_log_path': 'experiments/run/local-analytics/client_utilities.csv',
            'log_detailed': True
        }
        
        self.selector = OortSelector(
            selector_config,
            sample_seed=self.args.get('sample_seed', 233)
        )
        
        # Client information storage
        self.client_info = {}
        self.round_count = 0
        
        # Configuration
        self.model_size = self.args.get('model_size', 65536)  # in KB
        self.batch_size = self.args.get('batch_size', 32)
        self.upload_epoch = self.args.get('upload_epoch', 20)
        self.clock_factor = self.args.get('clock_factor', 1.0)
        
        logging.info("OortClientSampler initialized with full Oort implementation")
    
    def register_client(self, client_id, data_size, duration=2000, 
                       compute_speed=None, bandwidth=None):
        """
        Register client with system and data statistics
        
        Args:
            client_id: Unique client identifier
            data_size: Number of training samples
            duration: Initial duration estimate (seconds)
            compute_speed: Computation speed (samples/sec), optional
            bandwidth: Network bandwidth (KB/sec), optional
        """
        # Calculate initial reward (statistical utility based on data size)
        initial_reward = float(data_size)
        
        # Store client profile
        self.client_info[client_id] = {
            'data_size': data_size,
            'duration': duration,
            'compute_speed': compute_speed if compute_speed is not None else 1.0,
            'bandwidth': bandwidth if bandwidth is not None else 1000.0,
            'last_loss': None,
            'last_update_round': 0
        }
        
        # Register with Oort selector
        feedbacks = {
            'reward': initial_reward,
            'duration': duration
        }
        self.selector.register_client(client_id, feedbacks)
        
        logging.info(f"[Oort] Registered client {client_id}: "
                    f"data_size={data_size}, duration={duration:.2f}s, "
                    f"initial_reward={initial_reward:.2f}")
    
    def update_client(self, client_id, loss, duration, round_num, 
                     success=True, auxiliary_info=None):
        """
        Update client statistics after training round
        
        Args:
            client_id: Client identifier
            loss: Training loss (statistical utility indicator)
            duration: Actual training duration in seconds
            round_num: Current round number
            success: Whether training completed successfully
            auxiliary_info: Additional client metrics (optional dict)
        """
        if client_id not in self.client_info:
            logging.warning(f"[Oort] Client {client_id} not registered, skipping update")
            return
        
        # Calculate reward (statistical utility)
        # Reward = sqrt(loss) * data_size (as in original Oort)
        data_size = self.client_info[client_id]['data_size']
        reward = math.sqrt(loss) * data_size
        
        # Normalize duration to minutes
        duration_minutes = duration
        
        # Prepare feedback for selector
        feedbacks = {
            'reward': reward,
            'duration': duration_minutes,
            'time_stamp': round_num,
            'status': success
        }
        
        # Update selector
        self.selector.update_client_util(client_id, feedbacks)
        
        # Update local client info
        self.client_info[client_id]['last_loss'] = loss
        self.client_info[client_id]['duration'] = duration
        self.client_info[client_id]['last_update_round'] = round_num
        
        if auxiliary_info:
            self.client_info[client_id].update(auxiliary_info)
        
        logging.info(f"[Oort] Updated client {client_id}: "
                    f"loss={loss:.4f}, reward={reward:.2f}, "
                    f"duration={duration:.2f}s ({duration_minutes:.2f}min), "
                    f"success={success}")
    
    def update_duration(self, client_id, duration):
        """
        Update client duration separately
        
        Args:
            client_id: Client identifier
            duration: Duration in seconds
        """
        if client_id in self.client_info:
            self.client_info[client_id]['duration'] = duration
            self.selector.update_duration(client_id, duration)
    
    def select_clients(self, num_clients, feasible_clients=None, round_num=None):
        """
        Select clients for next training round
        
        Args:
            num_clients: Number of clients to select
            feasible_clients: List of available client IDs (None = all registered)
            round_num: Current round number (None = auto-increment)
            
        Returns:
            List of selected client IDs
        """
        if round_num is not None:
            self.round_count = round_num
        else:
            self.round_count += 1
        
        # Default to all registered clients if not specified
        if feasible_clients is None:
            feasible_clients = list(self.client_info.keys())
        
        # Ensure feasible_clients are registered
        feasible_clients = [
            c for c in feasible_clients 
            if c in self.client_info
        ]
        
        if len(feasible_clients) == 0:
            logging.warning("[Oort] No feasible clients available")
            return []
        
        # Select participants using Oort algorithm
        selected = self.selector.select_participants(
            num_clients, 
            feasible_clients, 
            self.round_count
        )
        
        logging.info(f"[Oort] Round {self.round_count}: "
                    f"Selected {len(selected)}/{num_clients} clients from "
                    f"{len(feasible_clients)} feasible clients")
        
        return selected
    
    def get_completion_time(self, client_id, batch_size=None, 
                           upload_epoch=None, model_size=None):
        """
        Estimate client completion time
        
        Args:
            client_id: Client identifier
            batch_size: Batch size (default: from config)
            upload_epoch: Number of local epochs (default: from config)
            model_size: Model size in KB (default: from config)
            
        Returns:
            Estimated completion time in seconds
        """
        if client_id not in self.client_info:
            return float('inf')
        
        batch_size = batch_size or self.batch_size
        upload_epoch = upload_epoch or self.upload_epoch
        model_size = model_size or self.model_size
        
        client = self.client_info[client_id]
        
        # Computation time: (data_size / batch_size) * upload_epoch / compute_speed
        # Simplified: 3.0 * batch_size * upload_epoch / compute_speed
        comp_time = 3.0 * batch_size * upload_epoch / float(client['compute_speed'])
        
        # Communication time: model_size / bandwidth
        comm_time = model_size / float(client['bandwidth'])
        
        return (comp_time + comm_time) * self.clock_factor
    
    def register_speed(self, client_id, compute_speed, bandwidth):
        """
        Register or update client system speed
        
        Args:
            client_id: Client identifier
            compute_speed: Computation speed (samples/sec)
            bandwidth: Network bandwidth (KB/sec)
        """
        if client_id in self.client_info:
            self.client_info[client_id]['compute_speed'] = compute_speed
            self.client_info[client_id]['bandwidth'] = bandwidth
            
            # Update estimated duration
            estimated_duration = self.get_completion_time(client_id)
            self.update_duration(client_id, estimated_duration)
    
    def is_client_active(self, client_id, current_time=None):
        """
        Check if client is active (for client availability traces)
        
        Args:
            client_id: Client identifier
            current_time: Current timestamp (optional)
            
        Returns:
            Boolean indicating if client is active
        """
        # This is a placeholder - implement based on availability traces
        # In original Oort, this checks against user traces
        return client_id in self.client_info
    
    def get_client_stats(self):
        """
        Get statistics for all clients
        
        Returns:
            Dict mapping client_id to statistics dict
        """
        stats = {}
        for client_id in self.client_info:
            client_metrics = self.selector.get_client_reward(client_id)
            if client_metrics:
                stats[client_id] = {
                    'reward': client_metrics['reward'],
                    'count': client_metrics['count'],
                    'duration': client_metrics['duration'],
                    'last_round': client_metrics['time_stamp'],
                    'data_size': self.client_info[client_id]['data_size'],
                    'last_loss': self.client_info[client_id].get('last_loss'),
                }
        return stats
    
    def get_median_reward(self):
        """Get median (mean) reward of feasible clients"""
        return self.selector.get_median_reward()
    
    def get_all_metrics(self):
        """Get all internal metrics from selector"""
        return self.selector.getAllMetrics()
    
    def get_exploration_stats(self):
        """
        Get current exploration/exploitation statistics
        
        Returns:
            Dict with exploration metrics
        """
        return {
            'exploration_rate': self.selector.exploration,
            'num_unexplored': len(self.selector.unexplored),
            'num_exploit': len(self.selector.exploitClients),
            'num_explore': len(self.selector.exploreClients),
            'round_threshold': self.selector.round_threshold,
            'blacklist_size': len(self.selector.blacklist)
        }
    
    def get_data_info(self):
        """Get dataset information"""
        total_samples = sum(c['data_size'] for c in self.client_info.values())
        return {
            'total_clients': len(self.client_info),
            'total_samples': total_samples
        }
    
    def reset_round_count(self):
        """Reset round counter"""
        self.round_count = 0
    
    def save_state(self, filepath):
        """
        Save sampler state for checkpointing
        
        Args:
            filepath: Path to save state
        """
        import pickle
        state = {
            'selector_state': self.selector.totalArms,
            'client_info': self.client_info,
            'round_count': self.round_count,
            'exploration': self.selector.exploration,
            'round_threshold': self.selector.round_threshold,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        logging.info(f"[Oort] Saved state to {filepath}")
    
    def load_state(self, filepath):
        """
        Load sampler state from checkpoint
        
        Args:
            filepath: Path to load state from
        """
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.selector.totalArms = state['selector_state']
        self.client_info = state['client_info']
        self.round_count = state['round_count']
        self.selector.exploration = state['exploration']
        self.selector.round_threshold = state['round_threshold']
        
        logging.info(f"[Oort] Loaded state from {filepath}")




