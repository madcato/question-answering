require 'daru'

require './openai_utils'
require './sqlite_utils'

### Load csv with Daru
data = Daru::DataFrame.from_csv("../train_tiny_list.csv")
# iterate through each row
data.each_row do |row|
  question = row[0]
  answer = row[1]
  embedding = $openai.generate_embedding(question)
  $sqlite.save_embedding(question, answer, embedding)
end
