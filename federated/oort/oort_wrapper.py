from .oort_selector import OortSelector
import logging
import math

class OortClientSampler:
    """Wrapper of Oort"""
    
    def __init__(self, args):
        self.args = args
        self.selector = OortSelector({
            'exploration_factor': 0.9,
            'exploration_decay': 0.95,
            'exploration_min': 0.2,
            'round_threshold': 10,
            'round_penalty': 2.0,
            'cut_off_util': 0.7
        })
        self.client_info = {}
        self.round_count = 0
        
    def register_client(self, client_id, data_size, duration=1.0):
        """Register client"""
        
        initial_reward = data_size
        feedbacks = {
            'reward': initial_reward,  
            'duration': duration
        }
        self.selector.register_client(client_id, feedbacks)
        self.client_info[client_id] = {
            'data_size': data_size,
            'duration': duration
        }
        logging.info(f"[Oort] Registered client {client_id}: "
                    f"data_size={data_size}, initial_reward={initial_reward}")
        
    def update_client(self, client_id, loss, duration, round_num):
        """Update client stats after training"""
        
        if client_id not in self.client_info:
            logging.warning(f"[Oort] Client {client_id} not registered, skipping update")
            return
        # Reward = statistical utility (loss-based)
        data_size = self.client_info[client_id]['data_size']
        reward = math.sqrt(loss) * data_size
        
        # Normalize duration to minutes
        duration_minutes = duration / 60.0
        
        feedbacks = {
            'reward': reward,
            'duration': duration_minutes,
            'time_stamp': round_num
        }
        self.selector.update_client_util(client_id, feedbacks)
        
        # Save loss for fallback
        self.client_info[client_id]['last_loss'] = loss
        
        logging.info(f"[Oort] Updated client {client_id}: "
                    f"loss={loss:.4f}, reward={reward:.4f}, "
                    f"duration={duration_minutes:.2f}min")
        
    def select_clients(self, num_clients, feasible_clients, round_num):
        """Select clients for next rounds"""
        
        self.round_count = round_num
        
        selected = self.selector.select_participants(
            num_clients, 
            feasible_clients, 
            round_num
        )
        logging.info(f"[Oort] Round {round_num}: Selected {len(selected)} clients")
        return selected
    
    def get_client_stats(self):
        """Get stats to debug"""
        stats = {}
        for client_id, info in self.selector.totalArms.items():
            stats[client_id] = {
                'reward': info['reward'],
                'count': info['count'],
                'duration': info['duration'],
                'last_round': info['time_stamp']
            }
        return stats