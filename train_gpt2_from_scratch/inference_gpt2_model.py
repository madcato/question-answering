from transformers import pipeline

qa_model = pipeline("text-generation", model="output")
answer = qa_model("<S> Does any know what is the lining of these containers? BPA free doesn't necessarily mean it's better. <SEP>", max_length=64)
print(answer)

answer = qa_model("<S> do you have to put by the window? can you put it where there is no window? <SEP>", max_length=64)
print(answer)

answer = qa_model("<S> Have these jimmies been pre-rustled? <SEP>", max_length=64)
print(answer)

answer = qa_model("<S> Is it dairy-free? <SEP>", max_length=64)
print(answer)
