import weaviate
import json

client = weaviate.Client("http://localhost:8080") # <== if you use Docker-compose

schema = client.schema.get()
print(json.dumps(schema))
