import os
import torch
import torch.nn as nn
from collections import OrderedDict

def get_state_dict(shard_num, prefix=None):
    d = torch.load(os.path.join(model_path, f"pytorch_model_{shard_num:05d}-of-00072.bin"))
    return d if prefix is None else OrderedDict((k.replace(prefix, ''), v) for k, v in d.items())

from transformers import AutoTokenizer, AutoModelForCausalLM, BloomConfig
from transformers.models.bloom.modeling_bloom import BloomBlock, build_alibi_tensor

model_path = "/hdd/gloom" # replace with your local folder path
config = BloomConfig.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = 'cpu'