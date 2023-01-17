import weaviate

client = weaviate.Client("http://localhost:8080")

# data_obj = {
#     "content": "Esta es una prueba de texto para ser convertida en un vector de embeddings"
# }

# data_uuid = client.data_object.create(
#   data_obj,
#   "Document"
# )

all_objects = client.data_object.get(class_name="Document", with_vector=True)
print(all_objects)