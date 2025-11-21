from .oort_selector import OortSelector
import logging

class OortClientSampler:
    """Wrapper để tích hợp Oort vào FedPylot"""
    
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
        
    def register_client(self, client_id, data_size, duration=1.0):
        """Đăng ký client vào Oort selector"""
        feedbacks = {
            'reward': data_size,  # Utility = data size
            'duration': duration
        }
        self.selector.register_client(client_id, feedbacks)
        self.client_info[client_id] = {
            'data_size': data_size,
            'duration': duration
        }
        
    def update_client(self, client_id, loss, duration, round_num):
        """Update client stats sau khi training"""
        # Reward = statistical utility (loss-based)
        reward = math.sqrt(loss) * self.client_info[client_id]['data_size']
        
        feedbacks = {
            'reward': reward,
            'duration': duration,
            'time_stamp': round_num
        }
        self.selector.update_client_util(client_id, feedbacks)
        
    def select_clients(self, num_clients, feasible_clients, round_num):
        """Chọn clients cho round tiếp theo"""
        selected = self.selector.select_participants(
            num_clients, 
            feasible_clients, 
            round_num
        )
        logging.info(f"[Oort] Round {round_num}: Selected {len(selected)} clients")
        return selected