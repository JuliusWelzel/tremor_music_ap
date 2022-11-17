library(tidyverse)
library(ggpubr)
library(rstatix)
library(stringr)
library(dplyr)

dt <- read.csv(("C:/Users/User/Desktop/kiel/tremor_music_ap/results/group_anova.csv"))

dt %>%
  group_by(visit, group) %>%
  get_summary_stats(Norm.amp, type = "mean_sd")

dt2 <- dt %>%
  filter(str_detect(task, "Sitting flexed"))


bxp <- ggboxplot(dt2, x = "group", y = "Norm.amp", color = "visit", palette = "jco" )
bxp

dt2 <- dt2[,c("id","group","visit","Norm.amp")]

res.aov <- anova_test(data = dt2, dv = Norm.amp, wid = id,within = c(visit),between = c(group))
get_anova_table(res.aov)
