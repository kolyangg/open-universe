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


## 11 May - updated ver (cross-attn only on Mel + FILM on latent)
from .condition_11May import ConditionerNetwork # second try to fix SIL masking
from .universe_gan_NS_11May import UniverseGAN  # upd guid_attn loss to add masking
from .textencoder_11May import TextEncoder
from .score import ScoreNetwork
## 11 May - updated ver (cross-attn only on Mel + FILM on latent)


