from transformers import pipeline

qa_model = pipeline("text2text-generation", model="mrm8488/spanish-t5-small-sqac-for-qa")
answer = qa_model("Tengo un problema con el sistema de archivos, parace que he perdido los persmisos para acceder a los ficheros A3 y A4. ¿Puedo hacer una copia de seguridad de los ficheros A3 y A4?", max_length=100)
print(answer)
