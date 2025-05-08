# Copyright 2024 LY Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The UNIVERSE++ and UNIVERSE models

Author: Robin Scheibler (@fakufaku)
"""
# # from .condition import ConditionerNetwork
# from .condition_NS_film import ConditionerNetwork ### NEW WITH TEXT ENCODER ###
# from .lora import UniverseLoRA
# from .mdn import MixtureDensityNetworkLoss
# from .score import ScoreNetwork
# from .universe import Universe
# # from .universe_gan import UniverseGAN
# # from .universe_gan_NS import UniverseGAN
# from .universe_gan_NS2 import UniverseGAN
# # from .textencoder_NS import TextEncoder
# from .textencoder_bert import TextEncoder


# from .condition import ConditionerNetwork
# from .condition_NS_film import ConditionerNetwork ### NEW WITH TEXT ENCODER ###
#from .condition_NS_film_ca import ConditionerNetwork ### NEW WITH TEXT ENCODER and CA ###
# from .condition_NS_plbert import ConditionerNetwork ### NEW WITH TEXT ENCODER and CA ###


# from .universe_gan import UniverseGAN
# from .universe_gan_NS import UniverseGAN


# from .textencoder_NS import TextEncoder
# from .textencoder_bert import TextEncoder
# from .textencoder_bert_ca import TextEncoder

# from .textencoder_plbert import TextEncoder




from .lora import UniverseLoRA
from .mdn import MixtureDensityNetworkLoss
# from .score import ScoreNetwork
# from .universe import Universe
# from .universe_NS_try import Universe
# from .universe_NS import Universe
# from .universe import Universe


# # ### PL-BERT SIMPLE
# from .condition_NS_plbert_simple import ConditionerNetwork ### Cross-attention only (no FILM)
# from .universe_gan_NS2 import UniverseGAN
# from .textencoder_plbert_op import TextEncoder

# # ### PL-BERT UPDATED
# from .condition_NS_plbert2 import ConditionerNetwork ### Cross-attention only (no FILM)
# from .universe_gan_NS2 import UniverseGAN
# from .textencoder_plbert_op2 import TextEncoder


# # DEFAULT
# from .score import ScoreNetwork
# # from .condition import ConditionerNetwork
# from .condition_check2 import ConditionerNetwork
# from .universe_gan_NS import UniverseGAN


# # DEFAULT TESTING - 17 APR
# from .score import ScoreNetwork
# # from .condition import ConditionerNetwork
# from .condition_check2 import ConditionerNetwork
# from .universe_gan_NS_check import UniverseGAN

# from .condition_NS_plbert_adj import ConditionerNetwork 
# from .universe_gan_NS2_adj import UniverseGAN

# # ### PL-BERT UPDATED - TEXT FEATURES IN MORE PLACES (06 APR)
# from .condition_NS_plbert3 import ConditionerNetwork ### Cross-attention only (no FILM)
# from .universe_gan_NS3 import UniverseGAN
# from .textencoder_plbert_op3 import TextEncoder
# from .score3 import ScoreNetwork

# # ### REVERTING OLD VERSION MB GOOD (08 APR)
# from .condition_NS_plbert import ConditionerNetwork
# from .universe_gan_NS2 import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # ### REVERTING OLD VERSION MB GOOD (09 APR) - TRYING TO FIX TRAINING PROCESS
# # from .condition_NS_plbert_adj import ConditionerNetwork 
# from .condition_NS_plbert_adj_clean import ConditionerNetwork 
# from .universe_gan_NS2_adj import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # # 10 Apr version = 09 Apr + new position for TextConditionoer (at end of ConditionerEncoder)
# # from .condition_NS_plbert_adj import ConditionerNetwork 
# from .condition_NS_plbert_adj_clean_ce import ConditionerNetwork 
# from .universe_gan_NS2_adj import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # # 11 Apr version: changing TextEncoder and CrossAttention dims to 512
# from .condition_NS_plbert_adj_clean_ce_512 import ConditionerNetwork 
# from .universe_gan_NS2_adj import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork

# # # 12 Apr version: 512 dims + two blocks for TextEncoder (initial + at end of CE)
# from .condition_NS_plbert_adj_clean_2x_512 import ConditionerNetwork 
# from .universe_gan_NS2_adj import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # 14 Apr version = 10 Apr + using 4 seconds of audio, filtering longer samples and padding shorter ones
# # from .condition_NS_plbert_adj_clean_ce_4s import ConditionerNetwork 
# from .condition_NS_plbert_adj_clean_ce_4s_norm import ConditionerNetwork 
# from .universe_gan_NS2_adj_4s import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # 15 Apr version = Fixing high grad issue in 14 Apr ver
# # from .condition_NS_plbert_adj_clean_ce_4s import ConditionerNetwork 
# from .condition_NS_plbert_adj_clean_ce_4s_fix import ConditionerNetwork 
# from .universe_gan_NS2_adj_4s import UniverseGAN
# from .textencoder_plbert_op_fix import TextEncoder
# from .score import ScoreNetwork


# # 16 Apr version = Fixing masks
# # from .condition_NS_plbert_adj_clean_ce_4s import ConditionerNetwork 
# from .condition_NS_plbert_adj_clean_ce_4s_fix import ConditionerNetwork 
# from .universe_gan_NS2_adj_4s2 import UniverseGAN
# from .textencoder_plbert_op_fix import TextEncoder
# from .score_4s import ScoreNetwork


# # 17 Apr version = Fixing grads
# # from .condition_NS_plbert_adj_clean_ce_4s import ConditionerNetwork 
# # from .condition_NS_plbert_adj_clean_ce_4s_fix import ConditionerNetwork 
# from .condition_NS_plbert_adj_clean_ce_4s_fix_norm import ConditionerNetwork 
# from .universe_gan_NS2_adj_4s2 import UniverseGAN
# from .textencoder_plbert_op_fix import TextEncoder
# from .score_4s import ScoreNetwork

# # # 17 Apr version2 = 10 Apr + simpler 4s dataset
# # from .condition_NS_plbert_adj import ConditionerNetwork 
# # from .condition_NS_plbert_adj_clean_ce import ConditionerNetwork 
# from .condition_NS_plbert_adj_clean_ce_check import ConditionerNetwork 
# from .universe_gan_NS2_adj import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # # 18 Apr version = 17 Apr + masking audio padding
# # from .condition_NS_plbert_adj import ConditionerNetwork 
# # from .condition_NS_plbert_adj_clean_ce import ConditionerNetwork 
# from .condition_NS_plbert_adj_clean_ce_check import ConditionerNetwork 
# # from .m_universe_gan_NS2_adj import UniverseGAN
# from .m_universe_gan_NS2_adj2 import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # # 19 Apr version = 18 Apr + adding text masking
# from .condition_18Apr import ConditionerNetwork 
# from .m_universe_gan_NS2_adj2 import UniverseGAN
# from .textencoder_plbert_op_fix import TextEncoder
# from .score import ScoreNetwork



# # # 27 Apr version = 512 dim for FILM and Cross-Atten
# from .condition_18Apr_512 import ConditionerNetwork 
# from .m_universe_gan_NS2_adj2 import UniverseGAN
# from .textencoder_plbert_op_fix import TextEncoder
# from .score import ScoreNetwork

# # # # 30 Apr version = 27 Apr + separate LR for text
# from .condition_30Apr_512 import ConditionerNetwork 
# from .m_universe_gan_NS2_adj3 import UniverseGAN
# from .textencoder_plbert_op_fix import TextEncoder
# from .score import ScoreNetwork


# # # # 01 May version - adding mask to ConditionerNetwork
# # from .condition_01May_512 import ConditionerNetwork 
# from .condition_01May_512_2x import ConditionerNetwork 
# from .m_universe_gan_NS2_adj4 import UniverseGAN
# # from .m_universe_gan_NS2_adj4_lgfix import UniverseGAN
# # from .textencoder_plbert_op_fix import TextEncoder
# from .textencoder_plbert_op_fix_norm import TextEncoder
# from .score import ScoreNetwork


# # # # 03 May version - trying to fix for no text case
# # from .condition_01May_512 import ConditionerNetwork 
# # from .condition_01May_512_2x import ConditionerNetwork 
# from .condition_01May_512_2x_map import ConditionerNetwork 
# # from .m_universe_gan_NS2_adj4_fix import UniverseGAN
# from .m_universe_gan_NS2_adj4_fix2 import UniverseGAN # fixing for no text case (LR scheduler)
# # from .m_universe_gan_NS2_adj4_lgfix import UniverseGAN
# # from .textencoder_plbert_op_fix import TextEncoder
# from .textencoder_plbert_op_fix_norm import TextEncoder
# from .score import ScoreNetwork


# # # # 04 May version - check vs original version
# # from .condition_orig_masked import ConditionerNetwork 
# from .condition_01May_512_2x_map import ConditionerNetwork 
# # from .m_universe_gan_NS2_adj4_fix2 import UniverseGAN # fixing for no text case (LR scheduler)
# # from .m_universe_gan_NS2_adj4_fix2_orig import UniverseGAN # fixing for no text case (LR scheduler)
# # from .universe_gan_NS_orig import UniverseGAN
# # from .universe_gan_NS_orig2 import UniverseGAN
# from .m_universe_gan_NS2_adj4_fix2_orig import UniverseGAN 
# from .textencoder_plbert_op_fix_norm import TextEncoder
# from .score import ScoreNetwork



# # 04 May version (almost same as before as no change vs original needed)
# from .condition_01May_512_2x_map import ConditionerNetwork 
# # from .condition_04May import ConditionerNetwork  # add POS embedding
# from .universe_gan_NS_04May import UniverseGAN # add guid_attn loss
# from .textencoder_04May import TextEncoder
# from .score import ScoreNetwork


# # 05 May version
# # from .condition_05May_nopos import ConditionerNetwork # no POS embedding yet
# from .condition_05May_pos import ConditionerNetwork # try POS embedding
# from .universe_gan_NS_05May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_04May import TextEncoder
# from .score import ScoreNetwork


# # 06 May version
# from .condition_06May import ConditionerNetwork # try POS embedding
# from .universe_gan_NS_06May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_04May import TextEncoder
# from .score import ScoreNetwork



# # 06 May version v2 (fixing attention further)
# from .condition_06May2 import ConditionerNetwork # try POS embedding
# from .universe_gan_NS_06May2 import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_04May import TextEncoder
# from .score import ScoreNetwork


# 08 May version - further improvements to attention
from .condition_08May import ConditionerNetwork # try POS embedding
from .universe_gan_NS_08May import UniverseGAN  # upd guid_attn loss to add masking
from .textencoder_08May import TextEncoder
from .score import ScoreNetwork




# # # Trying WavLM
# # from .condition_NS_plbert_adj import ConditionerNetwork 
# from .condition_wv import ConditionerNetwork 
# # from .condition_check import ConditionerNetwork 
# from .universe_gan_NS2_adj import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # 20 Apr - trying WavLM (v2)
# from .condition_wv2 import ConditionerNetwork 
# # from .condition_check import ConditionerNetwork 
# from .m_universe_gan_NS2_adj2 import UniverseGAN
# from .textencoder_plbert_op_fix import TextEncoder
# from .score import ScoreNetwork




# # ### PL-BERT UPDATED - TEXT FEATURES AFTER GRU (CURRENT)
# from .condition_NS_plbert5 import ConditionerNetwork ### Cross-attention only (no FILM)
# from .universe_gan_NS3 import UniverseGAN
# from .textencoder_plbert_op3 import TextEncoder
# from .score3 import ScoreNetwork


### PL-BERT SIMPLE (MIIPHER-LIKE)
# from .condition_NS_plbert_simple_m import ConditionerNetwork ### new Miipher-like easier version
# from .universe_gan_NS2_m import UniverseGAN ### new Miipher-like easier version
# from .textencoder_plbert_op import TextEncoder

### BERT SIMPLE
# from .condition_NS_plbert_simple import ConditionerNetwork ### new Miipher-like easier version
# from .universe_gan_NS2 import UniverseGAN
# from .textencoder_bert_new import TextEncoder

# ### BERT SIMPLE (MIIPHER-LIKE)
# from .condition_NS_plbert_simple_m import ConditionerNetwork ### new Miipher-like easier version
# from .universe_gan_NS2_m import UniverseGAN ### new Miipher-like easier version
# from .textencoder_bert_new import TextEncoder