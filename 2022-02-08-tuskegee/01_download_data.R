# Get the Data

# Read in with tidytuesdayR package 
# Install from CRAN via: install.packages("tidytuesdayR")
# This loads the readme and all the datasets for the week of interest

# Either ISO-8601 date or year/week works!
library(tidytuesdayR)
library(readr)
library(tidyverse)

# tuesdata <- tidytuesdayR::tt_load('2022-02-08')
tuesdata <- tidytuesdayR::tt_load(2022, week = 6)

airmen <- tuesdata$airmen
write_csv(airmen, "airmen.csv")

# Or read in the data manually

airmen <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2022/2022-02-08/airmen.csv')
