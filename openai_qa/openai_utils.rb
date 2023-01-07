require 'ruby/openai'

### OpenAI Embeddings
OPENAI_ACCESS_TOKEN=ENV['OPENAI_ACCESS_TOKEN']
EMBEDDINGS_SIZE=4096
MODEL_NAME = "curie"
DOC_EMBEDDINGS_MODEL = "text-search-#{MODEL_NAME}-doc-001"
QUERY_EMBEDDINGS_MODEL = "text-search-#{MODEL_NAME}-query-001"
COMPLETIONS_MODEL = "text-davinci-002"

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

  def generate_answer(prompt)
    response = @client.completions(parameters: { 
      prompt: prompt, 
      max_tokens: 300, 
      temperature: 0.0, 
      frequency_penalty: 0.0, 
      presence_penalty: 0.0,
      top_p: 1.0,
      model: COMPLETIONS_MODEL
    })
    
    if response.parsed_response['error'].nil? == false
      puts response.parsed_response['error']['message'] 
      exit
    end
    return response.parsed_response['choices'].first["text"]
  end
end

# Falta añadir un método para encontrar todos los más relevantes contextos y concatenerlos antes de hacer la petición de completar respuesta: mirar sección 4 de https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb
