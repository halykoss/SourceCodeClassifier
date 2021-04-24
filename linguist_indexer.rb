require 'rugged'
require 'linguist'
require 'find'
require 'thread/pool'
require 'thread'
require 'csv'

pathes = []
Find.find('source-code-set/') do |path|
  pathes << path unless FileTest.directory?(path)
end

num_split = 10

splitted_arr = pathes.each_slice( (pathes.size/num_split).round ).to_a

pool = Thread.pool(num_split)

splitted_arr.each do |obj|
    pool.process do
        obj.each do |value|
            puts value + "," + Linguist::FileBlob.new(value).language.name + "\n"
        end
    end
end

while not pool.done? do end

pool.shutdown
