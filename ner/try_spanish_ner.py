from turtle import mode
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

model_name = "Davlan/bert-base-multilingual-cased-ner-hrl"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = ["Mi nombre es Daniel Vela y vivo en Zaragoza",
          "Debo viajar a Madrid y llamar al tlf. 638820829",
          "Mi dirección es C/ San Juan de Dios, nº 1, Madrid"]

ner_results = nlp(example)
print(ner_results)