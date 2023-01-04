require 'ruby/openai'

OPENAI_ACCESS_TOKEN=ENV['OPENAI_ACCESS_TOKEN']

client = OpenAI::Client.new(access_token: OPENAI_ACCESS_TOKEN)

response = client.completions(engine: "davinci", parameters: { prompt: "Once upon a time story", max_tokens: 5 })
puts response.parsed_response['error']['message'] if response.parsed_response['error'].nil? == false
puts response.parsed_response['choices'].map{ |c| c["text"] }
