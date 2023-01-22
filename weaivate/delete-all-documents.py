import weaviate
import json

client = weaviate.Client("http://localhost:8080")

client.data.delete_all()