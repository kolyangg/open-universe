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


# ## 11 May - updated ver (cross-attn only on Mel + FILM on latent)
# # from .condition_11May2 import ConditionerNetwork # second try to fix SIL masking
# from .condition_11May2m import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_11May import TextEncoder
# from .score import ScoreNetwork
# ## 11 May - updated ver (cross-attn only on Mel + FILM on latent)


# ## 12 May - 11May ver + adding SIL token and not masking spaces
# from .condition_11May2_silsp import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_11May_silsp import TextEncoder
# from .score import ScoreNetwork
# ## 12 May - 11May ver + adding SIL token and not masking spaces


# ## 13 May - 11May_m + WavLM (with add loss)
# from .condition_11May2m_wv import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_11May_silsp import TextEncoder
# from .score import ScoreNetwork
# ## 13 May - 11May_m + WavLM (with add loss)



# ## 14 May - try 11May_m + xphonebert for text encoder
# # from .condition_11May2 import ConditionerNetwork # second try to fix SIL masking
# from .condition_11May2m import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_14May_xph import TextEncoder
# from .score import ScoreNetwork
# ## 14 May - try 11May_m + xphonebert for text encoder


# ## 15 May - 11May_m + WavLM (without WVLM loss) + upd WavLMAdapter
# from .condition_11May2m_wv2 import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_11May_silsp import TextEncoder
# from .score import ScoreNetwork
# ## 15 May - 11May_m + WavLM (without WVLM loss) + upd WavLMAdapter


# ## 15 May - 11May + MFA loss
# from .condition_11May2_silsp_tg import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May_tg import UniverseGAN  # upd guid_attn loss to add masking
# # from .textencoder_14May_xph2_tg import TextEncoder
# from .textencoder_14May_xph2_tg2 import TextEncoder
# from .score import ScoreNetwork
# ## 15 May - 11May + MFA loss


# ## 16 May - 11May_m + WavLM (with WVLM loss) + upd WavLMAdapter
# from .condition_11May2m_wv2 import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May_wvloss import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_11May_silsp import TextEncoder
# from .score import ScoreNetwork
# ## 16 May - 11May_m + WavLM (with WVLM loss) + upd WavLMAdapter


# ## 16 May - 11May (ful_film_sp) + XPhoneBERT
# from .condition_11May2_silsp import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_14May_xph2 import TextEncoder
# from .score import ScoreNetwork
# ## 16 May - 11May (ful_film_sp) + XPhoneBERT


# ## 16 May - 11May (ful_film_sp) + XPhoneBERT + WavLM (with WVLM loss)
# from .condition_11May2_silsp_wv import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_11May_wvloss import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_14May_xph2 import TextEncoder
# from .score import ScoreNetwork
# ## 16 May - 11May (ful_film_sp) + XPhoneBERT + WavLM (with WVLM loss)


## 19 May - 11May_m + WavLM DOUBLE (with WVLM loss) + upd WavLMAdapter (based on 16 May wv+full_film_m2_wvloss)
from .condition_11May2m_wv2_double import ConditionerNetwork
from .universe_gan_NS_11May_wvloss import UniverseGAN  # Now using wavlm-large embds for loss function (but using wavlm-base-plus-sv model) - TBC
from .textencoder_11May_silsp import TextEncoder
from .score import ScoreNetwork
## 19 May - 11May_m + WavLM DOUBLE (with WVLM loss) + upd WavLMAdapter (based on 16 May wv+full_film_m2_wvloss)


# ## 12 May - updated ver
# # from .condition_11May2 import ConditionerNetwork # second try to fix SIL masking
# from .condition_12May import ConditionerNetwork # second try to fix SIL masking
# from .universe_gan_NS_12May import UniverseGAN  # upd guid_attn loss to add masking
# from .textencoder_11May import TextEncoder
# from .score import ScoreNetwork
# ## 12 May - updated ver


