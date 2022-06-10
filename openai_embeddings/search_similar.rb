require './openai_utils'
require './sqlite_utils'

### Search text
QUESTION_FROM_TRAIN1 = '23 bucks for 1 package?'
QUESTION_FROM_TRAIN2 = '27 dollars for one package?'

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