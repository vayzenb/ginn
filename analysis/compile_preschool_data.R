
rm(list = ls())                                                                  #reset environment


getwd()

library('ggplot2')
library('plyr')
library('readr')
library('tidyverse')


setwd('C:/Users/vayze/Desktop/GitHub_Repos/ginn/analysis')
#setwd('C:/Users/User/OneDrive - UNT System/CMU Stuff/faceperc')


load("face_data.RData")
compiled_subs = all_subs
stim_data = basic_acc

#file_route <- fs::dir_ls('preschoolperc')                                          #load in all participant data
#pre_part <- read.csv(file = 'part_sheets/kid_percdata.csv')                                 # read participant sheet
#avg_acc = matrix(0, 11, 3)                                                      # create matrix
#pre_part = as.data.frame(pre_part)


subs = unique(compiled_subs$participant)
all_subs = NULL
for (ss in subs){
  print(ss)
  curr_sub = compiled_subs[compiled_subs$participant == ss,]
  curr_sub$rt_stand =scale(as.numeric(curr_sub$key_resp.rt), center = TRUE, scale =TRUE)
  curr_sub$rt_stand[abs(curr_sub$rt_stand) >= 2.5] = NA
  
  
  all_subs = rbind(all_subs, curr_sub) #combine into one big dataframe
  
}


for (qq in 1:nrow(stim_data)){
  stim_data[qq,3] = mean(all_subs$key_resp.corr[all_subs$stim1 == stim_data$stim1[qq] & 
                                                  all_subs$stim2 == stim_data$stim2[qq] |
                                                        all_subs$stim2 == stim_data$stim1[qq] & 
                                                        all_subs$stim1 == stim_data$stim2[qq]], na.rm = TRUE)
  
  
  # same thing u just did for acc but for rt 
  stim_data[qq,4] = mean(all_subs$key_resp.rt[all_subs$stim1 == stim_data$stim1[qq] & 
                                                  all_subs$stim2 == stim_data$stim2[qq] |
                                                  all_subs$stim2 == stim_data$stim1[qq] & 
                                                  all_subs$stim1 == stim_data$stim2[qq]], na.rm = TRUE)
  
  stim_data[qq,5] = mean(all_subs$rt_stand[all_subs$stim1 == stim_data$stim1[qq] & 
                                               all_subs$stim2 == stim_data$stim2[qq] |
                                               all_subs$stim2 == stim_data$stim1[qq] & 
                                               all_subs$stim1 == stim_data$stim2[qq]], na.rm = TRUE)
  
  
}





all_subs = NULL
for (rr in 1:nrow(pre_part)){                                                     # sort data by participants  
  
  newpart = read.csv2(file_route[rr], header= TRUE,sep= ',')                    # read csv
  newpart = newpart[6:nrow(newpart),]                                            # distinguishing that we only want row 7 & onward
  newpart$rt_stand = scale(as.numeric(newpart$key_resp.rt), center = TRUE, scale =TRUE) #standardize RTs
  newpart$rt_stand[newpart$rt_stand > 2.5] <- NA
  newpart$rt_stand = abs(newpart$rt_stand[newpart$rt_stand < 2.5]) #exclude outliers
  newpart = as.data.frame(newpart)
  newpart$key_resp.rt = as.numeric(newpart$key_resp.rt)
  
  all_subs = rbind(newpart, all_subs) #combine into one big dataframe
  
}





###
##### looping every image pair in the dataset
###

# creating the matrix and looping through to match pics
basic_acc = matrix(0, nrow(im_pair),5)

for (qq in 1:nrow(im_pair)){                                                       # if the image pair is the same as in the participant sheet,

#assigning the first 2 columns to be stims 1 and 2  
  basic_acc[qq,1] = im_pair$IM1[qq]
  basic_acc[qq,2] = im_pair$IM2[qq]

# assigning 3rd to be acc, as long as pic from long dataframe is same as pic from face_pairs.csv
# AND as long as the 2 images arent the save
     basic_acc[qq,3] = mean(all_subs$key_resp.corr[which(all_subs$stim1 ==
                              im_pair$IM1[qq] & all_subs$stim2 == im_pair$IM2[qq] |
                                all_subs$stim2 == im_pair$IM1[qq] & 
                                all_subs$stim1 == im_pair$IM2[qq])], na.rm = TRUE)
     
     
# same thing u just did for acc but for rt 
     basic_acc[qq,4] = mean(all_subs$key_resp.rt[which(all_subs$stim1 ==
                                                            im_pair$IM1[qq] & all_subs$stim2 == im_pair$IM2[qq] |
                                                            all_subs$stim2 == im_pair$IM1[qq] & 
                                                            all_subs$stim1 == im_pair$IM2[qq])], na.rm = TRUE)
     
     basic_acc = as.data.frame(basic_acc)
     

#### same thing but for standardized rxn time
     basic_acc[qq,5] = mean(all_subs$rt_stand[which(all_subs$stim1 ==
                                                            im_pair$IM1[qq] & all_subs$stim2 == im_pair$IM2[qq] |
                                                            all_subs$stim2 == im_pair$IM1[qq] & 
                                                            all_subs$stim1 == im_pair$IM2[qq])], na.rm = TRUE)
 
     colnames(basic_acc) = c('stim1', 'stim2', 'acc', 'resp_time','standardrt')             #column & row names
     basic_acc$acc = as.numeric(basic_acc$acc)
     basic_acc$resp_time = as.numeric(basic_acc$resp_time)
     basic_acc$standardrt = as.numeric(basic_acc$standardrt)
    
}


--------------------------------------------------------------------------------------------------

##########
##########  accuracy histogram  
##########  
  
  
#
#counting all the times that each accuracy occured
#
count = basic_acc %>%
  count(acc)                #counting acc and makign it numeric 
  count$acc = as.numeric(count$acc)
  
  count$n = as.numeric(count$n)
  as.data.frame(count)
  count = na.omit(count)
  
#binning the accuracies
count$bin = cut(count$acc,c(0.2,0.4,0.6,0.8,1))
table(count$bin)
colnames(count) = c('acc', 'occur', 'binacc')


# accuracy histogram
#

accgraph = ggplot(data = count) +                                             # establishing graph
  
  geom_col(aes(x=acc, y = occur))


------------------------------------------------------------------------------------------------------
###########  
########### STANDARDIZED response time (z score) histogram 
########### 
  
  
#
# this is counting all the times that each z score occured, first by:
  
# set up cut-off values 
breaks <- c(0,.5,1,1.5,2,2.5)


# specify interval/bin labels

z <- c("[0-.5)","[.5-1)", "[1-1.5)", "[1.5-2)", "[2-2.5)")

# bucketing values into bins
zcount <- cut(all_subs$rt_stand, 
              breaks=breaks, 
              include.lowest=TRUE, 
              right=FALSE, 
              labels=z)
zcount = data.frame(summary(zcount))
zcount$zscore = row.names(zcount)
colnames(zcount) = c('occurances', 'zscore')
zcount$zscore <- factor(zcount$zscore, levels = zcount$zscore[order(zcount$zscore)])


# z score histogram
#

accgraph = ggplot(data = zcount) +                                             # establishing graph
  
  geom_col(aes(x=zscore, y = occurances)) +
  scale_x_discrete(labels = c("[0-.5)", "[.5-1)","[1-1.5)", "[1.5-2)", "[2-2.5)"))






--------------------------------------------------------------------------------------------------------

########
####### NONSTANDARDIZED response time histogram
#######

# set up cut-off values 
breaks <- c(1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8)

# specify interval/bin labels

tags <- c("[1-1.5)","[1.5-2)", "[2-2.5)", "[2.5-3)", "[3-3.5)", "[3.5-4)","[4-4.5)", "[4.5-5)","[5-5.5)","[5.5-6)","[6-6.5)", "[6.5-7)","[7-7.5)","[7.5-8)")


# bucketing values into bins
bin_rt <- cut(basic_acc$resp_time, 
                  breaks=breaks, 
                  include.lowest=TRUE, 
                  right=FALSE, 
                  labels=tags)
bin_rt = as.data.frame(summary(bin_rt))
bin_rt$time = row.names(bin_rt)
colnames(bin_rt) = c('Occurances', 'Time')                         #column & row names


##graphing response time
ggplot(data = bin_rt) +                                             # establishing graph
geom_col(aes(x=Time, y = Occurances)) + 
scale_x_discrete(labels = c('1    ','','2     ','','3     ','','4     ','','5     ','','6     ','','7     ','','8')) 


------------------------------------------------------------------------------------------------------------
  
######### outdated, alt route to all_subs (percpicdata) of using 1 big data frame, without z scores
######### this works but no longer using it, reference later as a way to make 1 big data frame   
  
perclist <-list.files('C:/Users/User/OneDrive - UNT System/CMU Stuff/faceperc/preschoolperc')                                                                              ##listing all the files and then naming the list
percpicdata = ldply(perclist, read_csv)                                         # upload all data into 1 big matrix
percpicdata = as.data.frame(percpicdata)                                       # turn into data frame
im_pair <- read.csv(file = 'part_sheets/face_pairs.csv')                                    # read image pair sheet
  
  