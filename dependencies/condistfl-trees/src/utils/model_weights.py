# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Dict

import torch


def extract_weights(model: torch.nn.Module) -> Dict:
    """Extract model weights as a dictionary of numpy arrays."""
    return {k: v.cpu().numpy() for k, v in model.state_dict().items()}


def load_weights(model: torch.nn.Module, weights: Dict) -> torch.nn.Module:
    """Load weights into a model. Converts numpy arrays to tensors if needed."""
    import numpy as np
    
    # Convert numpy arrays and scalars to tensors
    state_dict = {}
    model_state = model.state_dict()
    
    for k, v in weights.items():
        # Check if it's a numpy array or numpy scalar
        if isinstance(v, (np.ndarray, np.generic)):
            # Convert to tensor
            tensor_v = torch.from_numpy(np.asarray(v))
            
            # Special handling for num_batches_tracked (should be int64)
            if 'num_batches_tracked' in k:
                tensor_v = tensor_v.long()
            
            # Match dtype of model parameter if it exists
            elif k in model_state:
                tensor_v = tensor_v.to(dtype=model_state[k].dtype)
            
            state_dict[k] = tensor_v
        else:
            state_dict[k] = v
    
    model.load_state_dict(state_dict)
    return model
