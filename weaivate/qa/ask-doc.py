import weaviate

client = weaviate.Client("http://bolt:8080")

ask = {
  "question": "What is the meaning of the circles of the olympic flag?",
  "properties": ["content"]
}

result = (
  client.query
  .get("Document", ["content", "_additional {answer {hasAnswer certainty property result startPosition endPosition} }"])
  .with_ask(ask)
  .with_limit(10)
  .do()
)

documents = result['data']['Get']['Document']

for document in documents:
  result = document["_additional"]["answer"]["result"]
  print(result)