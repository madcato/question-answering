import weaviate

client = weaviate.Client("http://localhost:8080")

schema = {
  "classes": [
    {
      "class": "Document",
      "description": "A class called document",
      "moduleConfig": {
        "text2vec-transformers": {
          "poolingStrategy": "masked_mean",
          "vectorizeClassName": False
        }
      },
      "properties": [
        {
          "dataType": [
            "text"
          ],
          "description": "Content that will be vectorized",
          "moduleConfig": {
            "text2vec-transformers": {
              "skip": False,
              "vectorizePropertyName": False
            }
          },
          "name": "content"
        }
      ],
      "vectorizer": "text2vec-transformers"
    }
  ]
}

client.schema.create(schema)