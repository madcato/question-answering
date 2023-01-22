import weaviate
import json

client = weaviate.Client("http://localhost:8080")

all_objects = client.data_object.get(class_name="Document", with_vector=False)

for obj in all_objects['objects']:
   client.data_object.delete(obj['id'])
