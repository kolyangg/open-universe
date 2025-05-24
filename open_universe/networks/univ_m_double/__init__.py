from .lora import UniverseLoRA
from .mdn import MixtureDensityNetworkLoss


## 19 May - 11May_m + WavLM DOUBLE (with WVLM loss) + upd WavLMAdapter (based on 16 May wv+full_film_m2_wvloss)
from .condition_11May2m_wv2_double import ConditionerNetwork
from .universe_gan_NS_11May_wvloss import UniverseGAN  # Now using wavlm-large embds for loss function (but using wavlm-base-plus-sv model) - TBC
from .textencoder_11May_silsp import TextEncoder
from .score import ScoreNetwork
## 19 May - 11May_m + WavLM DOUBLE (with WVLM loss) + upd WavLMAdapter (based on 16 May wv+full_film_m2_wvloss)



