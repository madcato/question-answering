import torch
from transformers import BertModel, BertTokenizer
# original model engllish language 
# model_name = 'bert-base-uncased'
model_name = 'dccuchile/bert-base-spanish-wwm-uncased'

tokenizer = BertTokenizer.from_pretrained(model_name)
# load
model = BertModel.from_pretrained(model_name)
input_text1 = "Hay que poner 34gr de azucar"
input_text2 = "Poner 26gr de edulcorante en polvo"
input_text3 = "Ir a pescar en verano es una buena idea"
# tokenizer-> token_id
input_ids = tokenizer.encode(input_text1, add_special_tokens=True)
input_ids = torch.tensor([input_ids])
with torch.no_grad():
    output1 = model(input_ids)[0]
output1 = output1.mean(1)[0]

# tokenizer-> token_id
input_ids = tokenizer.encode(input_text2, add_special_tokens=True)
input_ids = torch.tensor([input_ids])
with torch.no_grad():
    output2 = model(input_ids)[0]
output2 = output2.mean(1)[0]

# tokenizer-> token_id
input_ids = tokenizer.encode(input_text3, add_special_tokens=True)
input_ids = torch.tensor([input_ids])
with torch.no_grad():
    output3 = model(input_ids)[0]
output3 = output3.mean(1)[0]



cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

cosine12 = cos(output1, output2)
cosine23 = cos(output2, output3)
cosine13 = cos(output1, output3)

print("cosine12: ", cosine12)
print("cosine23: ", cosine23)
print("cosine13: ", cosine13)

# ---------------------------------------
# print(output)
# print(output.shape)
# print("NMorm")
# n = torch.linalg.norm(output)
# print("Norm:\n", n)

# normalized = torch.nn.functional.normalize(output,  p=2.0, dim = 0)
# n = torch.linalg.norm(normalized)
# print(n)