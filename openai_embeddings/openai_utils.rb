require 'ruby/openai'

### OpenAI Embeddings
OPENAI_ACCESS_TOKEN=ENV['OPENAI_ACCESS_TOKEN']
EMBEDDINGS_SIZE=1024

class OpenAI_API
  def initialize
    @client = OpenAI::Client.new(access_token: OPENAI_ACCESS_TOKEN)
  end

  def generate_embedding(text)
    response = @client.embeddings(engine: "text-similarity-ada-001", parameters: { input: text })
    data = response.parsed_response['data'].first
    return data['embedding']
  end
end

$openai = OpenAI_API.new