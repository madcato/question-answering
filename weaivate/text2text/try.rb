require 'weaviate'

client = Weaviate::Client.new(url: 'http://tesla:8080')

# Creating a new data object class in the schema
p client.schema.create(
    class_name: 'Question',
    description: 'Information from a Jeopardy! question',
    properties: [
        {
            "dataType": ["text"],
            "description": "The question",
            "name": "question"
        }, { 
            "dataType": ["text"],
            "description": "The answer",
            "name": "answer"
        }, {
            "dataType": ["text"],
            "description": "The category",
            "name": "category"
        }
    ],
    # Possible values: 'text2vec-cohere', 'text2vec-openai', 'text2vec-huggingface', 'text2vec-transformers', 'text2vec-contextionary', 'img2vec-neural', 'multi2vec-clip', 'ref2vec-centroid'
    vectorizer: "text2vec-cohere"
)

# Get a single class from the schema
client.schema.get(class_name: 'Question')

# Get the schema
p client.schema.list()

# Remove a class (and all data in the instances) from the schema.
client.schema.delete(class_name: 'Question')

# Update settings of an existing schema class.
# Does not support modifying existing properties.
client.schema.update(
    class_name: 'Question',
    description: 'Information from a Wheel of Fortune question'
)

# Adding a new property
client.schema.add_property(
    class_name: 'Question',
    property: {
        "dataType": ["boolean"],
        "name": "homepage"
    }
)

# Inspect the shards of a class
client.schema.shards(class_name: 'Question')