import weaviate

## IMPORTANT: first run ./load-db.py

client = weaviate.Client("http://bolt:8080")

result = (
  client.query
  .get("Document", ["content", "_additional {certainty distance}"]) # certainty only supported if distance==cosine
  .with_hybrid("What is the meaning of the circles of the olympic flag?", alpha=1.0)  # alpha 0.5 means 50% of the result is from the sparse, 50% from the vector
  .with_limit(10)
  .do()
)

result = result['data']['Get']['Document']
print(result)
print(len(result))
print(result[0])