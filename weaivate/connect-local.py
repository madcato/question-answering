import weaviate
import json

client = weaviate.Client("http://bolt:8080") # <== if you use Docker-compose

schema = client.schema.get()
print(json.dumps(schema))
