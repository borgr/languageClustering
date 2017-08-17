library(cluster)
library(RColorBrewer)

## Now color the inner dendrogram edges
color_dendro <- function(node, colormap){
  if(is.leaf(node)){
    nodecol <- colormap$color[match(attr(node, "label"), colormap$Species)]
    attr(node, "nodePar") <- list(pch = NA, lab.col = nodecol)
    attr(node, "edgePar") <- list(col = nodecol)
  }else{
    spp <- attr(node, "label")
    dominantspp <- levels(spp)[which.max(tabulate(spp))]
    edgecol <- colormap$color[match(dominantspp, colormap$Species)]
    attr(node, "edgePar") <- list(col = edgecol)
  }
  return(node)
}

## Duplicate rownames aren't allowed, so we need to set the "labels"
## attributes recursively. We also label inner nodes here. 
rectify_labels <- function(node, df){
  newlab <- df[unlist(node, use.names = FALSE)]
  attr(node, "label") <- (newlab)
  return(node)
}

plot_diana <- function(filename){
filename <- basename(filename)
dir <- 'distance_metrics/'
clust.dir <- "clusters/"
my.data <- read.csv(paste(dir, filename, sep=""), header = T, row.names = 1)
my.dist <- as.dist(my.data)
real.clusters <-read.csv(paste(clust.dir, filename, sep=""), header = T, row.names = 1)
clusters.factor <- factor(real.clusters$group)
clust <- diana(my.dist)
dnd <- as.dendrogram(clust)

dnd <- dendrapply(dnd, rectify_labels, df = clusters.factor)

## Create a color palette as a data.frame with one row for each spp
# irisspp <- as.character(unique(iris$Species))
# iris.colormap <- data.frame(Species = irisspp, color = rainbow(n = length(irisspp)))
# iris.colormap[, 2] <- c("red", "blue", "green")
uniqspp <- as.character(unique(clusters.factor))
colormap <- data.frame(Species = uniqspp, color = rainbow(n = length(uniqspp)))
# colormap[, 2] <- c("red", "blue", "green")
colormap[, 2] <- c( "blue", "green", "red", "cadetblue1", "deeppink", "lightsalmon4","gray1","khaki1", "lightcoral", "chartreuse", "aquamarine", "chocolate", "azure3", "darkorange" ,"coral2", "blueviolet", "darkgoldenrod", "brown", "darkgreen", "deepskyblue")
colormap

dnd.colored <- dendrapply(dnd, color_dendro, colormap = colormap)

## Plot the dendrogram
distance.name <- tools::file_path_sans_ext(filename)
imagename <- tools::file_path_sans_ext(filename)
imagename <- paste(imagename, ".svg", sep = "")
print(paste("plotting to ",paste("plots/", imagename, sep = "")))
png(paste("plots/", imagename, sep = ""), width=10,height=10,units="cm",res=900, pointsize=2)
# svg(filename=paste("plots/", imagename, sep = ""), width=1024, height=800)
plot(dnd.colored, main = distance.name)
dev.off()
}

metric.paths <- list.files("distance_metrics", pattern="*.csv", full.names = TRUE)
res <-lapply(metric.paths, plot_diana)

