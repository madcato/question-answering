import weaviate

client = weaviate.Client("http://localhost:8080")

ask = {
  "question": "What is the olympic motto?",
  "properties": ["content"]
}

result = (
  client.query
  .get("Document", ["content", "_additional {answer {hasAnswer certainty property result startPosition endPosition} }"])
  .with_ask(ask)
  .with_limit(1)
  .do()
)

result = result['data']['Get']['Document'][0]["_additional"]["answer"]["result"]
print(result)