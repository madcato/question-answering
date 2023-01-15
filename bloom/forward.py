def forward(input_ids):
    # 1. Create attention mask and position encodings
    attention_mask = torch.ones(len(input_ids)).unsqueeze(0).bfloat16().to(device)
    alibi = build_alibi_tensor(input_ids.shape[1], config.num_attention_heads,
                               torch.bfloat16).to(device)
    # 2. Load and use word embeddings
    embeddings, lnorm = load_embeddings()
    hidden_states = lnorm(embeddings(input_ids))
    del embeddings, lnorm

    # 3. Load and use the BLOOM blocks sequentially
    for block_num in range(70):
        load_block(block, block_num)
        hidden_states = block(hidden_states, attention_mask=attention_mask, alibi=alibi)[0]
        print(".", end='')
    
    hidden_states = final_lnorm(hidden_states)
    
    #4. Load and use language model head
    lm_head = load_causal_lm_head()
    logits = lm_head(hidden_states)

    # 5. Compute next token 
    return torch.argmax(logits[:, -1, :], dim=-1)