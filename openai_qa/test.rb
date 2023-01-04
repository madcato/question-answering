require 'ruby/openai'

COMPLETIONS_MODEL = "text-davinci-002"
OPENAI_ACCESS_TOKEN=ENV['OPENAI_ACCESS_TOKEN']
client = OpenAI::Client.new(access_token: OPENAI_ACCESS_TOKEN)

def search(client, prompt)
  client.completions(parameters: {
  prompt: prompt, 
  temperature: 0,
  max_tokens: 300,
  top_p: 1,
  frequency_penalty: 0,
  presence_penalty: 0,
  model: COMPLETIONS_MODEL })
end

prompt = "Who won the 2020 Summer Olympics men's high jump?"

response = search(client, prompt)
p response if response.parsed_response['error'].nil? == false
puts response.parsed_response['choices'].map{ |c| c["text"] }
# Wrong response: "The 2020 Summer Olympics men's high jump was won by Mariusz Przybylski of Poland."

## Preventing hallucination with prompt engineering

prompt = "Answer the question as truthfully as possible, and if you're unsure of the answer, say \"Sorry, I don't know\"

Q: Who won the 2020 Summer Olympics men's high jump?
A:"

response = search(client, prompt)
p response if response.parsed_response['error'].nil? == false
puts response.parsed_response['choices'].map{ |c| c["text"] }
