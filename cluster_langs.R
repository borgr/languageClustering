require(cluster)
require(phangorn)
require(ggplot2)
require(ggmap)
require('Rtsne')

setwd("~/clojuresrc/phono-data/resources")
kurd.data <- read.csv('binary_df_pure.csv', sep = '\t', h = T)

# Hierarchical clustering

kurd.matrix <- kurd.data[,5:]
kurd.matrix <- kurd.data[,-c(1,2,3,4)]
rownames(kurd.matrix) <- kurd.data[,1]
kurd.dist <- dist(kurd.matrix, method = "manhattan")
three.clusters <- diana(kurd.dist, 3)
plot(as.phylo(as.hclust(three.clusters)))

# PAM clustering and maps

sbbox <- make_bbox(lon = kurd.data$Lon, lat = kurd.data$Lat)
my_map <- get_map(location = sbbox, maptype = 'toner')
kurd.clustering.3 = pam(kurd.dist, 3)
kurd.data['three_clusters'] = as.factor(kurd.clustering.3$clustering)
cairo_pdf('three_clusters.pdf', width = 11, height = 8)
ggmap(my_map) + geom_point(data = kurd.data,
                           mapping = aes(Lon, Lat,
                                         shape = three_clusters,
                                         colour = Group))
dev.off()

# tSNE analysis

kurd.matrix.no.dups <- kurd.matrix[-which(duplicated(kurd.matrix)),]
tsne <- Rtsne(kurd.matrix.no.dups, dims = 2, perplexity = 10)
plot(tsne$Y, t = 'n')
text(tsne$Y, labels=rownames(kurd.matrix.no.dups))
