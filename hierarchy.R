require(cluster)
suppressPackageStartupMessages(library(dendextend))
require('dendextend')

color_branches_by_clusters <- function(dend, clusters, unique_colors, order_value = FALSE, ...) {
  if(length(clusters) != length(labels(dend))) stop("clusters must be of the same length as the labels in dend.")
  
  if(order_value) clusters <- clusters[order.dendrogram(dend)]
  
  clusters <- factor(clusters)
  if(missing(unique_colors)) unique_colors <- levels(clusters)
  clusters <- as.numeric(clusters)
  
  branches_attr_by_clusters(dend, clusters = clusters, values = unique_colors)
}

metric.paths <- list.files("distance_metrics", pattern="*.csv", full.names = TRUE)
cluster.paths<- list.files("clusters", pattern="*.csv", full.names = TRUE)
my.files <- lapply(metric.paths, read.csv, header = T, row.names = 1)
my.clusters <- lapply(cluster.paths, read.csv, header = T, row.names = 1)
my.dists <-lapply(my.files, as.dist)
res <-lapply(my.files, diana)
tuples <-mapply(list, my.clusters, res, SIMPLIFY=F)
for (tuple in tuples){
  clust.names <- tuple[1]
  clust <- tuple[2]
  dnd <- as.dendrogram(clust)
  dnd %>% color_branches_by_clusters(as.numeric(clust.names$group),  c( "blue", "green", "red"), order_value = TRUE) %>% plot
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
my.data <- read.csv(paste(dir, filename, sep=""), header = T, row.names = 1)
filename <- 'inv_yole.csv'
my.dist <- as.dist(my.data)
real.clusters <-read.csv(paste(clust.dir, filename, sep=""), header = T, row.names = 1)
clustered <- diana(my.dist)
dnd <- as.dendrogram(clustered)


temp_col <- c( "blue", "green", "red", "cadetblue1", "deeppink", "lightsalmon4","gray1","khaki1", "lightcoral", "chartreuse", "aquamarine", "chocolate", "azure3", "darkorange" ,"coral2", "blueviolet", "darkgoldenrod", "brown", "darkgreen", "deepskyblue", "darkolivegreen1")[as.numeric(real.clusters$group)]
temp_col <- temp_col[order.dendrogram(dnd)]
temp_col <- factor(temp_col, unique(temp_col))

dnd %>% color_branches(clusters = as.numeric(temp_col), col = levels(temp_col)) %>% 
  set("labels_colors", as.character(temp_col)) %>% 
  plot

dnd %>% color_branches_by_clusters(as.numeric(real.clusters$group), col = levels(temp_col) , order_value = TRUE) %>% plot

