from transformers import pipeline

text2text_generator = pipeline("text2text-generation", model = "bigscience/T0")

response = text2text_generator("Is the word 'table' used in the same meaning in the two previous sentences? Sentence A: you can leave the books on the table over there. Sentence B: the tables in this book are very hard to read." )
## [{"generated_text": "No"}]
print(response)

response = text2text_generator("A is the son's of B's brother. What is the family relationship between A and B?")
## [{"generated_text": "brother"}]
print(response)

response = text2text_generator("Is this review positive or negative? Review: this is the best cast iron skillet you will ever buy")
## [{"generated_text": "positive"}]
print(response)

response = text2text_generator("Reorder the words in this sentence: justin and name bieber years is my am I 27 old.")
##  [{"generated_text": "Justin Bieber is my name and I am 27 years old"}]
print(response)