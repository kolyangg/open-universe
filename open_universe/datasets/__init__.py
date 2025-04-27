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
# Datasets module

Contains the necessary API to load and process data during training.

Author: Robin Scheibler (@fakufaku)
"""
# from .datamodule import DataModule
# from .static_dataset import NoisyDataset
# from .static_dataset_NS2 import NoisyDataset

# from .static_dataset_NS_4sec_simple import NoisyDataset

# from .datamodule_adj import DataModule
# from .static_dataset_NS_4sec_fix import NoisyDataset

# from .datamodule_adj2 import DataModule
# from .static_dataset_NS_4sec_fix2 import NoisyDataset

# 18 Apr: Variable batch length, total sec length constsant
# from .datamodule_adj3 import DataModule
# from .static_dataset_NS_4sec_fix3 import NoisyDataset

# # 18 Apr: Variable batch length, total sec length constsant + masking
# from .m_datamodule_adj3 import DataModule
# from .m_static_dataset_NS_4sec_fix3 import NoisyDataset

# 19 Apr: old 4 sec batches (filter out longer) with zero padding for small samples + masking
# from .m_datamodule_pad import DataModule
# from .m_static_dataset_NS_4sec_pad import NoisyDataset

# from .m_datamodule_pad2 import DataModule
# from .m_static_dataset_NS_4sec_pad2 import NoisyDataset


# 19 Apr: Combined version of all three options

# from .datamodule_combo import DataModule
# from .static_dataset_combo import NoisyDataset



# 27 Apr: Combined version + strfix

from .datamodule_combo import DataModule
from .static_dataset_combo_strfix import NoisyDataset
