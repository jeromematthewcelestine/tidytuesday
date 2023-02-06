library(tidyverse)
library(ggplot2)
library(readr)
library(maps)

cats <- read_csv("cats_uk.csv")
cats
summary(cats)


# mapping
UK <- map_data(map = "world", region = "UK")
?map_data

ggplot(cats) + geom_point(show.legend = FALSE,
                          aes(x = location_long,
                              y = location_lat,
                              color = as_factor(tag_id)),
                          size = 0.1) +
 geom_path(data = UK %>% filter(long< -3, lat < 51), aes(x = long, y = lat, group = group))
??maps
nrow(cats)

#
cats %>% group_by(tag_id) %>% summarize(count = n())

amber <- cats %>% filter(tag_id == "Amber-Tag")

ggplot(amber %>% filter(location_long > -4.65)) + 
  geom_point(aes(x = location_long, y = location_lat, color = timestamp), size = 3.0)

glimpse(cats)



