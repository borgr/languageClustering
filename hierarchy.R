require(cluster)


temp <- list.files("clustering/distance_metrics", pattern="*.csv", full.names = TRUE)
my.files <- lapply(temp, read.csv, header = T, row.names = 1)
my.dists <-lapply(my.files, as.dist)
res <-lapply(my.files, diana)
res <-lapply(my.files, plot)

dir <- 'clustering/distance_metrics/'
filename <- 'aligned.csv'
filename <-'bag_cosine.csv'
filename <-'bigram_cosine.csv'
filename <-'inv_hamming.csv'
filename <-''
filename <-''
my.data <- read.csv(paste(dir, filename, sep=""), header = T, row.names = 1)
my.dist <- as.dist(my.data)
plot(diana(my.dist))
