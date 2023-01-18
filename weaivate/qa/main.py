import weaviate

## IMPORTANT: first run ./load-db.py

client = weaviate.Client("http://localhost:8080")

result = (
  client.query
  .get("Document", ["content", "_additional {certainty distance}"]) # certainty only supported if distance==cosine
  .with_hybrid("What is the olympic motto?", alpha=1.0)  # alpha 0.5 means 50% of the result is from the sparse, 50% from the vector
  .do()
)

result = result['data']['Get']['Document']
print(result)
print(len(result))
print(result[0])