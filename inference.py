from transformers import pipeline

qa_model = pipeline("question-answering")
question = "Where do I live?"
context = "My name is Merve and I live in İstanbul."
qa_model(question = question, context = context)
## {'answer': 'İstanbul', 'end': 39, 'score': 0.953, 'start': 31}