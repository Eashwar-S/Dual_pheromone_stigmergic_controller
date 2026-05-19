"""Communication bandwidth tracking utility for swarm simulations."""
from typing import Dict, List


class CommunicationTracker:
    """Track communication bandwidth metrics for swarm coordination."""
    
    def __init__(self, message_size_bytes: int = 5):
        """
        Initialize communication tracker.
        
        Args:
            message_size_bytes: Size of each message in bytes (default: 5)
                - Position (x, y): 2 bytes
                - Status (failure): 1 byte
                - Header ID + Message type: 2 bytes
        """
        self.message_size_bytes = message_size_bytes
        self.total_messages = 0
        self.peak_messages_per_step = 0
        self.messages_per_step: List[int] = []
    
    def record_step(self, messages_this_step: int):
        """
        Record messages sent in current timestep.
        
        Args:
            messages_this_step: Number of messages sent this timestep
        """
        self.total_messages += messages_this_step
        self.peak_messages_per_step = max(self.peak_messages_per_step, messages_this_step)
        self.messages_per_step.append(messages_this_step)
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get bandwidth metrics.
        
        Returns:
            Dictionary containing:
                - total_messages: Total messages across all timesteps
                - total_bandwidth_bytes: Total data transmitted in bytes
                - peak_messages_per_step: Maximum messages in single timestep
                - peak_bandwidth_bytes: Maximum bandwidth in single timestep (bytes)
                - avg_messages_per_step: Average messages per timestep
                - avg_bandwidth_per_step: Average bandwidth per timestep (bytes)
        """
        num_steps = len(self.messages_per_step)
        avg_messages = self.total_messages / num_steps if num_steps > 0 else 0.0
        
        return {
            'total_messages': self.total_messages,
            'total_bandwidth_bytes': self.total_messages * self.message_size_bytes,
            'peak_messages_per_step': self.peak_messages_per_step,
            'peak_bandwidth_bytes': self.peak_messages_per_step * self.message_size_bytes,
            'avg_messages_per_step': avg_messages,
            'avg_bandwidth_per_step': avg_messages * self.message_size_bytes,
        }
    
    def reset(self):
        """Reset all tracking metrics."""
        self.total_messages = 0
        self.peak_messages_per_step = 0
        self.messages_per_step = []
