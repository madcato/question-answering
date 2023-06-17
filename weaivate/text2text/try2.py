from transformers import pipeline

text2text_generator = pipeline("text2text-generation")
response = text2text_generator("question: What is 42 ? context: 42 is the answer to life, the universe and everything")
[{'generated_text': 'the answer to life, the universe and everything'}]
print(response)

response = text2text_generator("translate from English to French: I'm very happy")
[{'generated_text': 'Je suis très heureux'}]
print(response)