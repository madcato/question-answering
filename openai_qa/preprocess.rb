# We have hosted the processed dataset, so you can download it directly without having to recreate it.
# This dataset has already been split into sections, one row for each section of the Wikipedia page.
require 'dotenv'
Dotenv.load
require 'daru'
require "numo/narray"
require './openai_utils'

### Load csv with Daru
data = Daru::DataFrame.from_csv('https://cdn.openai.com/API/examples/data/olympics_sections_text.csv')

## Daru set index
data = data.set_index(["title", "heading"])
data = data.first(500)

puts "#{data.size} rows in the data."

# data = data.first(5)

puts
puts

# Embeddings

openai = OpenAI_API.new(DOC_EMBEDDINGS_MODEL, QUERY_EMBEDDINGS_MODEL)

## Generate embeddings for each row of the daru dataframe
context_embeddings = openai.compute_doc_embedding(data)

p "context_embeddings.size #{context_embeddings.size}"

## Add embeddings to the dataframe
data.add_vector(:embedding, context_embeddings)
p data.head

def vector_similarity(x, y)
  x = Numo::NArray.asarray(x)
  y = Numo::NArray.asarray(y)
  return x.dot(y)
end

def order_document_sections_by_query_similarity(openai, query, context)
  query_embedding = openai.generate_query_embedding(query)
  contextmapsim = context.map_rows do |row|
    vector_similarity(query_embedding, row[:embedding])
  end
  p contextmapsim
  contextmapsim
end

context_similarity = order_document_sections_by_query_similarity(openai, "What is the olympic motto?", data)
data.add_vector(:similarity, context_similarity)

data.sort!([:similarity], ascending: false)

p data.head

puts "Respuesta: " + data['content'][0]