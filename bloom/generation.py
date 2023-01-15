input_sentence = "The SQL command to extract all the users whose name starts with A is: "
input_ids = tokenizer.encode(input_sentence, return_tensors='pt').to(device)
max_tokens = 10
for i in range(max_tokens): 
    print(f"Token {i + 1} ", end='')
    new_id = forward(input_ids)
    input_ids = torch.cat([input_ids, new_id.unsqueeze(-1)], dim=-1)
    print(tokenizer.decode(new_id))

print(tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True))
