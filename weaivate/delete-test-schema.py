import weaviate
import json

client = weaviate.Client("http://localhost:8080")

client.schema.delete_all()

schema = client.schema.get()
print(json.dumps(schema))