require './openai_utils'
require './sqlite_utils'

### Search text
QUESTION_FROM_TRAIN1 = '¿Cuanto cuesta un kilo de sal?'
QUESTION_FROM_TRAIN2 = 'Son 27 dolares por paquete'

def search(question)
  embedding = $openai.generate_embedding(question)
  answer = $sqlite.search(embedding)
end

answer1 = search(QUESTION_FROM_TRAIN1)
puts "Question 1: #{QUESTION_FROM_TRAIN1}"
puts "Answer 1: #{answer1} "

puts 
puts 

answer2 = search(QUESTION_FROM_TRAIN2)
puts "Question 2: #{QUESTION_FROM_TRAIN2}"
puts "Answer 2: #{answer2} "

puts 
puts