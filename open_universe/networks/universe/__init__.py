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
# from .condition_NS_plbert_simple import ConditionerNetwork ### Cross-attention only (no FILM)
from .condition_NS_plbert_simple_m import ConditionerNetwork ### new Miipher-like easier version
from .lora import UniverseLoRA
from .mdn import MixtureDensityNetworkLoss
from .score import ScoreNetwork
from .universe import Universe
# from .universe_gan import UniverseGAN
# from .universe_gan_NS import UniverseGAN
# from .universe_gan_NS2 import UniverseGAN
from .universe_gan_NS2_m import UniverseGAN ### new Miipher-like easier version
# from .textencoder_NS import TextEncoder
# from .textencoder_bert import TextEncoder
# from .textencoder_bert_ca import TextEncoder
# from .textencoder_plbert import TextEncoder
from .textencoder_plbert_op import TextEncoder