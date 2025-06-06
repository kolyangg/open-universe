### WAVLM 
from .lora import UniverseLoRA
from .mdn import MixtureDensityNetworkLoss


# # DEFAULT
from .score import ScoreNetwork
# from .condition_wv4 import ConditionerNetwork
from .condition_wv4_double_plus import ConditionerNetwork # 3 MAY - TESTING NEW VERSION (GENERIC + SPEAKER FEATURES)
# from .condition_wv4_single_plus import ConditionerNetwork # 4 MAY - TESTING PLUS VER OF SINGLE OPTION
# from .condition_wv4_single import ConditionerNetwork # 4 MAY - RESTESTING SINGLE BASE OPTION
# from .condition_wv4_large import ConditionerNetwork # 4 MAY - RESTESTING SINGLE LARGE OPTION (DEFAULT NOW)
from .universe_gan_NS_wv_loss import UniverseGAN