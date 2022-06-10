require 'sqlite3'

### SQLite3
SQLITE_FILE = "openai_embeddings.sqlite3"

class SQLITE3_API
  def initialize(embeddings_size)
    @db = SQLite3::Database.new(SQLITE_FILE)
    
    @correlatives = (1..embeddings_size).to_a
    @correlatives.map! {|x| "v#{x}" }
    correlatives_str = @correlatives.join(" float, ") + " float"

    rows = @db.execute <<-SQL
      create table if not exists embeddings (
      question text,
      answer text,
      #{correlatives_str}
    );
    SQL
  end

  # def calcularte_square(embedding)
  #   Math.sqrt(embedding.reduce(0) {|sum, x| sum + x * x})
  # end

  def save_embedding(question, answer, embedding)
    @db.execute "insert into embeddings values (?, ?, #{embedding.to_a.join(',')})", question, answer
  end

  def search(embedding)
    sum = @correlatives.zip(embedding).map {|ab| ab.join(" * ")}.join(" + ")

    cosine_similarity_command = <<-SQL
      select answer, #{sum} as cosim from embeddings order by cosim desc LIMIT 1;
    SQL

    rows = @db.execute(cosine_similarity_command)
    return rows.first
  end
end

$sqlite = SQLITE3_API.new(EMBEDDINGS_SIZE)
