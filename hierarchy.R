require(cluster)


temp <- list.files("clustering/distance_metrics", pattern="*.csv", full.names = TRUE)
my.files <- lapply(temp, read.csv, header = T)
my.dists <-lapply(my.files, as.dist)
res <-lapply(my.files, diana)
res <-lapply(my.files, plot)


my.data <- read.csv('clustering/distance_metrics/aligned.csv', header = T, row.names = T)