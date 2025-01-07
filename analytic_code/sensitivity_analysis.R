atlas <- 'ReinforcementLearning'
cluster_save_dir = "/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/clustering/cluster_label"
pheno_case_1y <- read.csv(paste0(cluster_save_dir, '/', atlas, '/', 'after1y', '_case_cluster.csv'))
pheno_case_1y <- pheno_case_1y[, c('subjectkey', 'cluster_label')]
colnames(pheno_case_1y) <- c('subjectkey', 'cluster_label_1y')
pheno_case_2y <- read.csv(paste0(cluster_save_dir, '/', atlas, '/', 'after2y', '_case_cluster.csv'))
pheno_case_2y <- pheno_case_2y[, c('subjectkey', 'cluster_label')]
colnames(pheno_case_2y) <- c('subjectkey', 'cluster_label_2y')
pheno_case_merge <- merge(pheno_case_1y, pheno_case_2y, by='subjectkey')

same <- c()
diff <- c()
for (i in 1:nrow(pheno_case_merge)){
  if (pheno_case_merge[i,'cluster_label_1y'] + pheno_case_merge[i,'cluster_label_2y'] == 1){
    diff <- append(diff,pheno_case_merge[i,'subjectkey'] )
  }else{
    same <- append(same, pheno_case_merge[i,'subjectkey'] )
  }
}
print(length(same))
print(length(diff))


## become overweight and obesity in both years 
pheno_1y <- read.csv('/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/dnn/ABCD_phenotype_total_1years_become_overweight_10PS_stratified_partitioned_5fold.csv')
pheno_1y <- pheno_1y[,c('subjectkey', 'BMI_sds_change', 'BMI_change', 'weight_change', 'BMI_status_baseline', 'BMI_status_1year')]
colnames(pheno_1y) <- c('subjectkey', 'BMI_sds_change_1y', 'BMI_change_1y', 'weight_change_1y', 'BMI_status_baseline', 'BMI_status_1year')
pheno_1y_merge <- merge(pheno_1y,pheno_case_merge, by='subjectkey')
pheno_2y <- read.csv('/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/dnn/ABCD_phenotype_total_2years_become_overweight_10PS_stratified_partitioned_5fold.csv')
pheno_2y <- pheno_2y[,c('subjectkey', 'BMI_sds_change', 'BMI_change', 'weight_change', 'BMI_status_2year')]
colnames(pheno_2y) <- c('subjectkey', 'BMI_sds_change_2y', 'BMI_change_2y', 'weight_change_2y',  'BMI_status_2year')
pheno_2y_merge <- merge(pheno_2y,pheno_case_merge, by='subjectkey')
pheno_1y_2y_case_merge <- merge(pheno_1y_merge,pheno_2y_merge, by='subjectkey')

## become overweight in only 1 year
pheno_case_1y_removed <-data.frame()
for (i in 1:nrow(pheno_case_1y)){
  if (!(pheno_case_1y[i,'subjectkey'] %in% pheno_case_merge$subjectkey)){
    pheno_case_1y_removed <- rbind(pheno_case_1y_removed, pheno_case_1y[i,])
  }
}
pheno_case_only_1y <- merge(pheno_case_1y_removed, pheno_1y, by='subjectkey') # add status at 1y of children who are obese/overweight in 1 year but not obese/oeverweight in 2 year
pheno_case_only_1y <- merge(pheno_case_only_1y, pheno_2y, by='subjectkey')  # add status at 2y of children who are obese/overweight in 1 year but not obese/oeverweight in 2 year

## become overweight in only 1 year
pheno_case_2y_removed <-data.frame()
for (i in 1:nrow(pheno_case_2y)){
  if (!(pheno_case_2y[i,'subjectkey'] %in% pheno_case_merge$subjectkey)){
    pheno_case_2y_removed <- rbind(pheno_case_2y_removed, pheno_case_2y[i,])
  }
}
pheno_case_only_2y <- merge(pheno_case_2y_removed, pheno_1y, by='subjectkey')  # add status at 1y of children who are obese/overweight in 2 year but not obese/oeverweight in 1 year
pheno_case_only_2y <- merge(pheno_case_only_2y, pheno_2y, by='subjectkey')  # add status at 1y of children who are obese/overweight in 2 year but not obese/oeverweight in 1 year


## plotting median
case_case_median = c(0, median(pheno_1y_2y_case_merge[,'BMI_sds_change_1y']), median(pheno_1y_2y_case_merge[,'BMI_sds_change_2y']))
case_control_median = c(0, median(pheno_case_only_1y[,'BMI_sds_change_1y']), median(pheno_case_only_1y[,'BMI_sds_change_2y']))
control_case_median = c(0, median(pheno_case_only_2y[,'BMI_sds_change_1y']), median(pheno_case_only_2y[,'BMI_sds_change_2y']))
# plot
plot(case_case_median, type = 'o', col = 'red', ylim = c(0,0.8), xaxt = 'n', xlab = 'Time Point', ylab = 'delta BMI-sds', lwd=2) 
lines(case_control_median, type = 'o', col = 'blue', lwd=2) 
lines(control_case_median, type = 'o', col = 'green', lwd=2)
lines(c(0.2, 0.2, 0.2), lty=c('dashed'), col='black')  # reference line
axis(side=1,at=c(1,2,3),labels=c("baseline","baseline ~ 1year","baseline ~ 2year"))
legend("topright",legend=c("Case-Case","Case-Control", "Control-Case"),fill=c("red","blue",'green'),box.lty=0,cex=1.0)




## check clustering after sensitivity test
atlas <- 'Amygdala'
cluster_save_dir = "/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/clustering/cluster_label/sensitivity"
pheno_case_1y <- read.csv(paste0(cluster_save_dir, '/', atlas, '/', 'after1y', '_case_cluster.csv'))
pheno_case_1y <- pheno_case_1y[, c('subjectkey', 'cluster_label')]
colnames(pheno_case_1y) <- c('subjectkey', 'cluster_label_1y')
atlas <- 'Amygdala'
pheno_case_2y <- read.csv(paste0(cluster_save_dir, '/', atlas, '/', 'after2y', '_case_cluster.csv'))
pheno_case_2y <- pheno_case_2y[, c('subjectkey', 'cluster_label')]
colnames(pheno_case_2y) <- c('subjectkey', 'cluster_label_2y')
pheno_case_merge <- merge(pheno_case_1y, pheno_case_2y, by='subjectkey')

same <- c()
diff <- c()
for (i in 1:nrow(pheno_case_merge)){
  if (pheno_case_merge[i,'cluster_label_1y'] + pheno_case_merge[i,'cluster_label_2y'] == 1){
    diff <- append(diff,pheno_case_merge[i,'subjectkey'] )
  }else{
    same <- append(same, pheno_case_merge[i,'subjectkey'] )
  }
}
print(length(same))
print(length(diff))






############ Scratch 
## check clustering after sensitivity test
atlas <- 'Amygdala'
cluster_save_dir = "/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/clustering/cluster_label/sensitivity/case_control"
pheno_case_1y_a <- read.csv(paste0(cluster_save_dir, '/', atlas, '/', 'after1y', '_case_cluster.csv'))
pheno_case_1y_a <- pheno_case_1y_a[, c('subjectkey', 'cluster_label')]
colnames(pheno_case_1y_a) <- c('subjectkey', 'cluster_label_a')
atlas <- 'HarvardOxford_Subcortical'
cluster_save_dir = "/Users/wangheehwan/Desktop/CNN_for_BMI/paper/data/clustering/cluster_label/sensitivity/case_control"
pheno_case_1y_b <- read.csv(paste0(cluster_save_dir, '/', atlas, '/', 'after1y', '_case_cluster.csv'))
pheno_case_1y_b <- pheno_case_1y_b[, c('subjectkey', 'cluster_label')]
colnames(pheno_case_1y_b) <- c('subjectkey', 'cluster_label_b')
pheno_case_a_b_merge <- merge(pheno_case_1y_a, pheno_case_1y_b, by='subjectkey')
same <- c()
diff <- c()
for (i in 1:nrow(pheno_case_a_b_merge)){
  if (pheno_case_a_b_merge[i,'cluster_label_a'] + pheno_case_a_b_merge[i,'cluster_label_b'] == 1){
    diff <- append(diff,pheno_case_a_b_merge[i,'subjectkey'] )
  }else{
    same <- append(same, pheno_case_a_b_merge[i,'subjectkey'] )
  }
}
print(length(same))
print(length(diff))

