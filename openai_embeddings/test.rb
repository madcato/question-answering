require 'ruby/openai'

OPENAI_ACCESS_TOKEN="sk-gBWxeN4HR5DnskfalaLCwQTG3ukVFL1Smioqn28h"

client = OpenAI::Client.new(access_token: OPENAI_ACCESS_TOKEN)

response = client.completions(engine: "davinci", parameters: { prompt: "Once upon a time story", max_tokens: 5 })
puts response.parsed_response['choices'].map{ |c| c["text"] }
