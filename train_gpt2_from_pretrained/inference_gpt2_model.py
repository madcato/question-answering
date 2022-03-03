from transformers import pipeline

qa_model = pipeline("text-generation", model="output_pretrained")
answer = qa_model("<S> Does any know what is the lining of these containers? BPA free doesn't necessarily mean it's better. <SEP>", max_length=256)
print(answer)
