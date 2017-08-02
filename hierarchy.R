require(cluster)
suppressPackageStartupMessages(library(dendextend))
require('dendextend')

temp <- list.files("distance_metrics", pattern="*.csv", full.names = TRUE)
my.files <- lapply(temp, read.csv, header = T, row.names = 1)
my.dists <-lapply(my.files, as.dist)
res <-lapply(my.files, diana)
for (clust in res){
  dend <- as.dendrogram(clust)
  color_branches(dend, )
  plot(clust)
}
dir <- 'distance_metrics/'
filename <- 'aligned.csv'
filename <-'bag_cosine.csv'
filename <-'bigram_cosine.csv'
filename <-'inv_hamming.csv'
filename <-''
filename <-''
my.data <- read.csv(paste(dir, filename, sep=""), header = T, row.names = 1)
my.dist <- as.dist(my.data)
real.clusters <-read.csv("clusters", header = T, row.names = 1)
clustered <- diana(my.dist)
# dend <- colour_branches(???)
plot(dend)


dir <- 'distance_metrics/'
clust.dir <- "clusters/"
filename <- 'inv_yole.csv'
my.data <- read.csv(paste(dir, filename, sep=""), header = T, row.names = 1)
my.dist <- as.dist(my.data)
real.clusters <-read.csv(paste(clust.dir, filename, sep=""), header = T, row.names = 1)
clustered <- diana(my.dist)
dnd <- as.dendrogram(clustered)

iris2 <- iris[sample(x = 1:150,size = 50,replace = F),]
clust <- diana(iris2)
dnd <- as.dendrogram(clust)
dend %>% color_branches_by_clusters(as.numeric(iris2$5),  c( "blue", "green", "red"), order_value = TRUE) %>% plot
color_branches_by_clusters <- function(dend, clusters, unique_colors, order_value = FALSE, ...) {
  if(length(clusters) != length(labels(dend))) stop("clusters must be of the same length as the labels in dend.")
  
  if(order_value) clusters <- clusters[order.dendrogram(dend)]
  
  clusters <- factor(clusters)
  if(missing(unique_colors)) unique_colors <- levels(clusters)
  clusters <- as.numeric(clusters)
  
  branches_attr_by_clusters(dend, clusters = clusters, values = unique_colors)
}

dnd %>% color_branches_by_clusters(as.numeric(real.clusters$group),  c( "blue", "green", "red"), order_value = TRUE) %>% plot
