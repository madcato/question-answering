require 'ruby/openai'

### OpenAI Embeddings
OPENAI_ACCESS_TOKEN=ENV['OPENAI_ACCESS_TOKEN']
EMBEDDINGS_SIZE=4096

class OpenAI_API
  def initialize(doc_embedding_model, query_embedding_model)
    @client = OpenAI::Client.new(access_token: OPENAI_ACCESS_TOKEN)
    @doc_embedding_model = doc_embedding_model
    @query_embedding_model = query_embedding_model
  end

  def generate_embedding(text, model_name)
    response = @client.embeddings(parameters: { input: text, model: model_name })
    data = response.parsed_response['data'].first
    return data['embedding']
  end

  def generate_doc_embedding(text)
    return generate_embedding(text, @doc_embedding_model)
  end

  def generate_query_embedding(text)
    return generate_embedding(text, @query_embedding_model)
  end

  ### Generate embeddings for each row of the daru dataframe
  def compute_doc_embedding(data)
    return data.map_rows do |row|
      row_text = row['content'].gsub("\n", " ")
      generate_doc_embedding(row_text)
    end
  end
end