library(tidyr)
library(dplyr)
library(ggplot2)

df <- read.csv(file='df_base_env02_c.csv')
last <- ncol(df)
columns <- df[2:last]

df <- gather(data=df,
                    key=temp,
                    value=value,
                    names(columns),
                    factor_key=TRUE)

df$model <- sub("\\_.*", "", df$temp)
df$income_path <- sub(".*_(.+)_.*", "\\1", df$temp)
df$var <- sub(".*_(.+)", "\\1", df$temp)
df$temp <- NULL

df <- df[, c(1,3,4,5,2)]
names(df)[names(df) == 'X'] <- 'timestep'

# df <- df[ which(df$var!='savings'), ]

ggplot(data=df, aes(x=timestep, y=value, colour=var)) +
    geom_point() +
    facet_grid(vars(model), vars(income_path))
