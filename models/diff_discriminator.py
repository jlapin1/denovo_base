import torch as th
from torch import nn
import models.model_parts as mp

class Discriminator(nn.Module):
    def __init__(
        self,
        running_units=512,
        num_input_tokens=24,
        d=64,
        h=8,
        dropout=0,
        embed_type='preembed',
        prenorm=False,
        ffn_multiplier=2,
        depth=6,
        timestep_dimension=128,
    ):
        super(Discriminator, self).__init__()
        
        """Timestep embedding"""
        self.time_embed = nn.Sequential(
            nn.Linear(timestep_dimension, timestep_dimension),
            nn.SiLU(),
            nn.Linear(timestep_dimension, timestep_dimension)
        )

        """seq_emb"""
        self.seq_emb = nn.Embedding(num_input_tokens, running_units, padding_idx=self.NT)

        """Transformer blocks"""
        attention_dict = {
            'indim': running_units, 
            'd': d, 
            'h': h,
            'dropout': dropout,
            'alphabet': False,
        }
        ffn_dict = {
            'indim': running_units,
            'unit_multiplier': ffn_multiplier, 
            'dropout': dropout,
            'alphabet': False,
        }
        self.main = nn.ModuleList([
            mp.TransBlock(
                attention_dict, 
                ffn_dict, 
                norm_type='layer',
                prenorm=prenorm, 
                embed_type=embed_type,
                embed_indim=timestep_dimension,
                is_cross=False,
                kvindim=None,
            ) 
            for _ in range(depth)
        ])
