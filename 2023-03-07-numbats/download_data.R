
library(readr)
library(magrittr)

numbats <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2023/2023-03-07/numbats.csv')
write_csv(numbats, "numbats.csv")
