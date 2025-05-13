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

# # 22 Apr: Aligned versions

# from .datamodule_aligned import DataModule
# from .static_dataset_aligned import NoisyDataset


# # 27 Apr: Aligned versions

# from .datamodule_aligned_fix import DataModule
# from .static_dataset_aligned2 import NoisyDataset


# # 02 May: Aligned versions

# from .datamodule_aligned_fix import DataModule
# from .static_dataset_aligned3_fix import NoisyDataset


# 03 May: Simplied version of aligned

# from .datamodule_aligned_fix import DataModule
from .datamodule_aligned_13May import DataModule
from .static_dataset_aligned3_fix import NoisyDataset