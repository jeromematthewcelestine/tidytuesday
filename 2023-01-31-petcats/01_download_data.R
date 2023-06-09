library(tidyverse)
library(ggplot2)
library(readr)
library(tidytuesdayR)
library(here)

tuesdata <- tidytuesdayR::tt_load('2023-01-31')

cats_uk <- tuesdata$cats_uk
cats_uk_reference <- tuesdata$cats_uk_reference

write_csv(cats_uk, here("2023-01-31-petcats/data", "cats_uk.csv"))
write_csv(cats_uk_reference, here("2023-01-31-petcats/data", "cats_uk_reference.csv"))
