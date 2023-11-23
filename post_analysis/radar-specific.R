library(dplyr)
library(ggplot2)
library(stringr)
library(readxl)
library(fmsb)

# 
# file1 = '/Users/fiona/Library/CloudStorage/OneDrive-mail.bnu.edu.cn/Project/PAC_Individual_difference/result/stats/clustering/cluster_fingerprint/fingerprint/fingerprint_rest1_left_mean_cluster.csv'
# file2 = '/Users/fiona/Library/CloudStorage/OneDrive-mail.bnu.edu.cn/Project/PAC_Individual_difference/result/stats/clustering/cluster_fingerprint/fingerprint/fingerprint_rest1_right_mean_cluster.csv'
# 
# 
# data1 = read.table(file1, header=TRUE, sep=",") # left
# data2 = read.table(file2, header=TRUE, sep=",") # right


# Male-specific
#data1 = data.frame(Frontal = 2, Temporale = 0, Parietal = 0, Insular = 0, Limbic = 0, Occipital = 1, Subcortical = 0)
#data2 = data.frame(Frontal = 7, Temporale = 3, Parietal = 3, Insular = 0, Limbic = 1, Occipital = 0, Subcortical = 0)

# FaMale-specific
# data1 = data.frame(Frontal = 7, Temporale = 1, Parietal = 2, Insular = 0, Limbic = 0, Occipital = 1, Subcortical = 0)
# data2 = data.frame(Frontal = 1, Temporale = 1, Parietal = 1, Insular = 0, Limbic = 0, Occipital = 0, Subcortical = 0)
# 

# hcp-male
data1 = data.frame(Frontal = 3, Temporale = 1, Parietal = 2, Insular = 0, Limbic = 0, Occipital = 0, Subcortical = 0)
data2 = data.frame(Frontal = 0, Temporale = 0, Parietal = 0, Insular = 0, Limbic = 0, Occipital = 0, Subcortical = 0)




# 
#df1 = data1[,][2:18] # cluster-
#df2 = data2[,][2:18] # cluster-
df = rbind(data1,data2)
# row name
rownames(df) <- c('Intra-lobe', 'Inter-lobes')


#Create data: note in High school for Jonathan:
# data <- as.data.frame(matrix( sample( 2:20 , 10 , replace=T) , ncol=10))
# colnames(data) <- c("math" , "english" , "biology" , "music" , "R-coding", "data-viz" , "french" , "physic", "statistic", "sport" )
# df=data
# To use the fmsb package, I have to add 2 lines to the dataframe: the max and min of each topic to show on the plot!
data <- rbind(rep(8,7) , rep(0,7) , df)


# Check your data, it has to look like this!
# head(data)

# Custom the radarChart !

#pdf("/Users/fiona/Library/CloudStorage/OneDrive-mail.bnu.edu.cn/Project/Prediction/Article/Articles_Figs/EPS_RAW/Fig6/LI_BrainMeasures_PredCorr_Radar.pdf", width = 200, height = 200)

# colors_border=c( rgb(0.09019608,0.27450980,0.63529412,0.7) , rgb(0.7,0.5,0.1,0.7) )
# colors_in=c( rgb(0.09019608,0.27450980,0.63529412,0.2), rgb(0.7,0.5,0.1,0.2) )


colors_border=c( rgb(0.2,0.5,0.5,0.9), rgb(0.8,0.2,0.5,0.9) , rgb(0.7,0.5,0.1,0.9) , rgb(0.5,0.6,0.3,0.9))
colors_in=c( rgb(0.2,0.5,0.5,0.4), rgb(0.8,0.2,0.5,0.4) , rgb(0.7,0.5,0.1,0.4) ,rgb(0.5,0.6,0.3,0.4))




p <- radarchart( data  , axistype=1 , 
                 
                 #maxmin = FALSE,
                 maxmin = TRUE,
                 seg = 8, # 5个环，6条线，即5组分割，这个数字和下面的caxislabels=seq(-0.2,0.3,0.1)一定要对应上，不然图像显示会有问题（错位）
                 
                 #custom polygon
                 #pcol=rgb(0.2,0.5,0.5,0.9) , pfcol=rgb(0.2,0.5,0.5,0.5) , plwd=3 , 
                 pcol=colors_border , pfcol=colors_in , plwd=3 ,  # 87A2FB， rgb(135, 162, 251)
                 
                 #custom the grid
                 cglcol="grey", cglty=1, axislabcol="grey", caxislabels=seq(0,8,1), cglwd=0.8,
                 
                 #custom labels
                 vlcex=0.8
)

p

# add legend
legend(x=0.95, y=1.3, legend = rownames(data[-c(1,2),]), bty = "n", pch=20 , col=colors_in , text.col = "grey", cex=1.2, pt.cex=3)
