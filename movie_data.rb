# MovieData class
class MovieData

    # Returns a new MovieData instance
    def initialize(path, name=nil)
        @training_filename, @test_filename = name.nil? ? [File.join(path, 'u.data'), nil] : [File.join(path, "#{name.to_s}.base"), File.join(path, "#{name.to_s}.test")]
        @most_similar_cache = {}
        load_data
    end

    # Loads data from the training set
    def load_data

        @data = []
        File.open(@training_filename) do |file|
            file.each_line do |line|
                @data << line.split.map {|x| x.to_i}
            end
        end

        @movie_id_indices = {}
        @user_id_indices = {}
        @data.each do |item|
            user_id, movie_id, rating, timestamp = item
            @movie_id_indices[movie_id] = {} unless @movie_id_indices.has_key?(movie_id)
            @movie_id_indices[movie_id][user_id] = item
            @user_id_indices[user_id] = {} unless @user_id_indices.has_key?(user_id)
            @user_id_indices[user_id][movie_id] = item
        end

        @max_nr_ratings = 0
        @movie_id_indices.each_value {|v| @max_nr_ratings = [@max_nr_ratings, v.length].max}

    end

    # Returns the popularity of the movie
    def popularity(movie_id)
        @movie_id_indices[movie_id].length.to_f / @max_nr_ratings.to_f
    end

    # Returns the popularity list
    def popularity_list
        @movie_id_indices.sort_by {|k, v| popularity(k)}.map {|m, p| m}.reverse
    end

    # Returns the similarity of the two users
    def similarity(user1, user2)
        common_movie_ids = @user_id_indices[user1].keys & @user_id_indices[user2].keys
        return nil if common_movie_ids.empty?
        dp = 0
        l1 = 0
        l2 = 0
        common_movie_ids.each do |movie_id|
            rating1 = @user_id_indices[user1][movie_id][2]
            rating2 = @user_id_indices[user2][movie_id][2]
            dp += rating1 * rating2
            l1 += rating1 * rating1
            l2 += rating2 * rating2
        end
        dp.to_f / Math::sqrt(l1.to_f * l2.to_f)
    end

    # Returns most similar users of the user
    def most_similar(u)
        return @most_similar_cache[u] if @most_similar_cache.has_key?(u)
        user_id_and_similarity = []
        @user_id_indices.keys.each do |user_id|
            next if user_id == u
            s = similarity(u, user_id)
            user_id_and_similarity << [user_id, s] unless s.nil?
        end
        @most_similar_cache[u] = user_id_and_similarity.sort_by {|u, s| s}.map {|u, s| u}.reverse
        @most_similar_cache[u]
    end

    # Returns the rating by the user for the movie
    def rating(u, m)
        @user_id_indices[u].has_key?(m) ? @user_id_indices[u][m][2] : 0
    end

    # Returns the prediction of the rating by the user for the movie
    def predict(u, m)
        ratings = []
        most_similar(u).each do |user_id|
            ratings << @user_id_indices[user_id][m][2] if @user_id_indices[user_id].has_key?(m)
            break if ratings.length == 5
        end
        return nil if ratings.empty?
        ratings.inject(:+).to_f / ratings.length
    end

    # Returns movies that the user has rated
    def movies(u)
        @user_id_indices[u].keys
    end

    # Returns users that have rated the movie
    def viewers(m)
        @movie_id_indices[m].keys
    end

    # Runs the test 
    def run_test(k=nil)
        data = []
        File.open(@test_filename) do |file|
            file.each_line do |line|
                u, m, r, t = line.split.map {|x| x.to_i}
                p = predict(u, m)
                data << [u, m, r, p] unless p.nil?
                break if !k.nil? && data.length == k
            end
        end
        MovieTest.new(data)
    end

end


# MovieTest class
class MovieTest

    # Returns a new MovieTest class
    def initialize(data)
        @data = data
        @mean = 0
        @stddev = 0
        @rms = 0
        @data.each do |u, m, r, p|
            @mean += (p - r).abs
            @rms += (p - r)**2
        end
        @mean /= @data.length
        @rms = Math::sqrt(@rms / @data.length)
        @data.each do |u, m, r, p|
            @stddev += ((p - r).abs - @mean)**2
        end
        @stddev = Math::sqrt(@stddev / @data.length)
    end

    # Returns the mean of error
    def mean
        @mean
    end

    # Returns the standard deviation of error
    def stddev
        @stddev
    end

    # Returns the root mean square of error
    def rms
        @rms
    end

    # Returns an array containing the data
    def to_a
        @data.map {|x| x.dup}
    end

end


[:u1, :u2, :u3, :u4, :u5].each do |un|
    puts "========== Running test #{un} =========="
    movie_data = MovieData.new('ml-100k', un)
    movie_test = movie_data.run_test(100)
    puts "Mean: #{movie_test.mean}"
    puts "Standard deviation: #{movie_test.stddev}"
    puts "RMS: #{movie_test.rms}"
end

puts "========== Benchmarking prediction =========="
movie_data = MovieData.new('ml-100k')
t_begin = Time.now
(1..10000).each {movie_data.predict(1, 1)}
t_end = Time.now
puts "Time cost of each prediction: #{(t_end - t_begin) / 10000} seconds"
