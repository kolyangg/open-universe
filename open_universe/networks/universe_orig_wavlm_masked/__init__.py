### WAVLM 
from .lora import UniverseLoRA
from .mdn import MixtureDensityNetworkLoss


# # DEFAULT
from .score import ScoreNetwork
# from .condition_wv4 import ConditionerNetwork
from .condition_wv4_double import ConditionerNetwork # 3 MAY - TESTING NEW VERSION (GENERIC + SPEAKER FEATURES)
from .universe_gan_NS_wv_loss import UniverseGAN