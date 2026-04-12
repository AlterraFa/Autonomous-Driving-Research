import itertools
import random
from utils.messages.logger import Logger

class BalancedRoadSampler:
    from .data_loader import PolySystemLoader
    def __init__(self, loader_instance: PolySystemLoader, batch_size):
        self.logger = Logger()
        self.multimodal_indices = loader_instance.multimodal_idx
        self.unimodal_indices   = loader_instance.unimodal_idx
        self.batch_size         = batch_size
        
        if batch_size % 2 != 0:
            self.logger.ERROR("Batch size must be even for a 50/50 split", exit_code = 1)
            
        self.half_batch = batch_size // 2
        
        self.len_majority = max(len(self.multimodal_indices), len(self.unimodal_indices))
        self.num_batches  = self.len_majority // self.half_batch
        
        
    def __iter__(self):
        multi_shuffled = self.multimodal_indices[:]
        uni_shuffled   = self.unimodal_indices[:]
        random.shuffle(multi_shuffled)
        random.shuffle(uni_shuffled)

        multi_it = itertools.cycle(multi_shuffled)
        uni_it   = itertools.cycle(uni_shuffled)

        
        for _ in range(self.num_batches):
            batch = []

            batch.extend(itertools.islice(multi_it, self.half_batch))
            batch.extend(itertools.islice(uni_it, self.half_batch))
            
            
            random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return self.num_batches