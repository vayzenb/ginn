rm(list=ls())
library(ggplot2)

setwd('B:/home/vayzenbe/GitHub_Repos/GiNN/Results')

ModelType = c('Face','Object', 'Random')
stim = c('vggface2_fbf', 'ImageNet_Objects')
stimCat = c('Face', 'Object')
cond =c('Upright', 'Inverted')

cols = c('Model', 'Stim', 'Cond', 'Acc')

df.Summary <- data.frame(Model=character(),Stim=character(), Cond=character(),Acc = numeric(),CI = numeric(),
                 stringsAsFactors=TRUE) 

for (mm in ModelType){
  for (ss in stimCat) {
    for (cc in cond){
      df = read.table(paste(mm,'_', ss, '_', cc, ".csv", sep=""),header = FALSE, sep=",")
      
      colnames(df) = cols
      
      df.Summary = data.frame(Model = c(df.Summary$Model, df$Model[1]), 
                              Stim = c(df.Summary$Stim, df$Stim[1]),
                              Cond = c(df.Summary$Cond, df$Cond[1]),
                              Acc = c(df.Summary$Acc, mean(df$Acc, na.rm = TRUE)),
                              CI = c(df.Summary$CI, 2*sd(df$Acc, na.rm = TRUE)/sqrt(nrow(df))))
    }
    
    
  }
  
}

df.Summary$Model[df.Summary$Model == "Face"] = "Face Selectivity"
df.Summary$Model[df.Summary$Model == "Object"] = "General Selectivity"
df.Summary$Model[df.Summary$Model == "Random"] = "Blank Slate"
df.Summary$Cond = factor(df.Summary$Cond, levels = c("Upright", "Inverted"))

ggplot(df.Summary, aes(x= Stim, y = Acc, fill = Cond)) + 
  geom_bar(stat= "identity",  color = "black", size = .8, width = .81, position = position_dodge(.99)) + facet_grid( ~ Model) +
  geom_linerange(aes(ymin = Acc - CI, ymax= Acc + CI), position=position_dodge(.99), size = .9) + theme_classic()

ggsave(filename =  'figures/Inversion.png', plot = last_plot(), dpi = 300,width =6, height = 4)
