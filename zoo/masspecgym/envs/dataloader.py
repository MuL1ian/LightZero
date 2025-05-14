import torch
import random
from torch.utils.data import Dataset



# class DebugSpectrumDataset(Dataset):
#     def __init__(self, file_path="/Users/boyeniu/Desktop/nus_course/NUS_JAX/MassEnv/policy_model/debug_spectrum_embeds.pt"):
#         self.data = torch.load(file_path)

#         list_lengths = []
#         for key, value in self.data.items():
#             if isinstance(value, list):
#                 list_lengths.append(len(value))
#             elif isinstance(value, torch.Tensor) and len(value.shape) > 0:
#                 list_lengths.append(value.shape[0])
#         if len(set(list_lengths)) > 1:
#             print(f"Error: data fields have different lengths: {list_lengths}")

#         self.size = list_lengths[0] if list_lengths else 0
#         print(f"Loaded {self.size} samples")

#     def __len__(self):
#         return self.size

#     def __getitem__(self, idx):
#         result = {}
#         for key, value in self.data.items():
#             if isinstance(value, list):
#                 if idx < len(value):
#                     result[key] = value[idx]
#             elif isinstance(value, torch.Tensor) and len(value.shape) > 0:
#                 if idx < value.shape[0]:
#                     result[key] = value[idx]
#         return result

#     def random_sample(self):
#         idx = random.randint(0, self.size - 1)
#         return self.__getitem__(idx)

# dataset = DebugSpectrumDataset()
# print(dataset.random_sample())

import selfies as sf
# smiles = 'CC(=O)N[C@@H](CC1=CC=CC=C1)C2=CC(=CC(=O)O2)OC'
# benzene_sf = sf.encoder(smiles)

# print(benzene_sf)
test = '[C][C][=C][C][=CC]'
print(sf.split_selfies(test))