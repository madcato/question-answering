def load_embeddings():
    state_dict = get_state_dict(shard_num=1, prefix="word_embeddings_layernorm.")
    embeddings = nn.Embedding.from_pretrained(state_dict.pop('word_embeddings.weight'))
    lnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, dtype=torch.bfloat16)
    lnorm.load_state_dict(state_dict)
    return embeddings.to(device), lnorm.to(device)

def load_causal_lm_head():
    linear = nn.utils.skip_init(
        nn.Linear, config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16)
    linear.load_state_dict(get_state_dict(shard_num=1, prefix="word_embeddings."), strict=False)
    return linear.bfloat16().to(device)

def load_block(block_obj, block_num):
    block_obj.load_state_dict(get_state_dict(shard_num=block_num + 2, prefix=f"h.{block_num}."))
    block_obj.to(device)

final_lnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon, dtype=torch.bfloat16)
final_lnorm.load_state_dict(get_state_dict(shard_num=72, prefix="ln_f."))
final_lnorm.to(device)
block = BloomBlock(config, layer_number=1).bfloat16()