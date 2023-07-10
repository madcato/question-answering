from transformers import AutoTokenizer

text = "Nandu->"

# checkpoint = "facebook/blenderbot_small-90M"
checkpoint = "./models/checkpoint-175000"
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot_small-90M")
inputs = tokenizer(text, return_tensors="pt").input_ids

from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
outputs = model.generate(inputs, max_new_tokens=40, do_sample=True, top_k=0, top_p=0.0)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))