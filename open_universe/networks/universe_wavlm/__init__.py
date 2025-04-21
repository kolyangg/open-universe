### WAVLM 
from .lora import UniverseLoRA
from .mdn import MixtureDensityNetworkLoss


# # # Trying WavLM
# # from .condition_NS_plbert_adj import ConditionerNetwork 
# from .condition_wv import ConditionerNetwork 
# # from .condition_check import ConditionerNetwork 
# from .universe_gan_NS2_adj import UniverseGAN
# from .textencoder_plbert_op import TextEncoder
# from .score import ScoreNetwork


# # 20 Apr - trying WavLM (v2) - working option (wavlm-base, mel loss)
# from .condition_wv2 import ConditionerNetwork
# # from .m_universe_gan_NS2_adj2 import UniverseGAN

# from .textencoder_plbert_op_fix import TextEncoder
# from .score import ScoreNetwork

# # # check init version
# # from .condition import ConditionerNetwork  
# from .universe_gan_NS import UniverseGAN



# 20 Apr - trying WavLM (v3) - trying wavlm-large (wavlm-large) and WavLMLMOSLoss
from .condition_wv3 import ConditionerNetwork

from .textencoder_plbert_op_fix import TextEncoder
from .score import ScoreNetwork

# # check init version 
# from .universe_gan_NS import UniverseGAN
from .universe_gan_NS_wv_loss import UniverseGAN # try WavLM loss instead of Mel


