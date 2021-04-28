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

#puts "Recuperati" + pathes.size.to_s

splitted_arr = pathes.each_slice( (pathes.size/num_split).round ).to_a

#puts "Divisi in " + splitted_arr.size.to_s  + " array "

#puts "da"
#splitted_arr.each do |obj|
#    puts obj.size.to_s + " elementi"
#end

pool = Thread.pool(num_split)

splitted_arr.each do |obj|
    pool.process do
        obj.each do |value|
            begin
                puts value + "," + Linguist::FileBlob.new(value).language.name + "\n"
            rescue
            end
        end
    end
end

while not pool.done? do end

pool.shutdown
