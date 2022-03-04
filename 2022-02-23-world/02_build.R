library(tidyverse)
library(ggplot2)
library(RColorBrewer)
library(scales)
library(extrafont)

freedom <- read_csv("freedom.csv")

n_countries <- freedom %>%
  filter(year == 2004) %>%
  group_by(Region_Name) %>%
  summarize(n_countries = n())

pct_freedom <- freedom %>%
  group_by(Region_Name, year, Status) %>%
  summarize(count = n()) %>%
  group_by(Region_Name, year) %>%
  mutate(total_count = sum(count), pct = count / total_count) 

ggplot(pct_freedom) +
  geom_area(aes(x = year, y = pct, fill = fct_relevel(Status, c("F", "PF", "NF")))) +
  facet_grid( ~ Region_Name) +
  scale_x_continuous(breaks = c(1995, 2020), expand = expansion(0)) +
  scale_y_continuous(labels = scales::percent, expand = expansion(0))  + 
  labs(title = "Percentage of Countries Classified as Free, Partially Free, and Not Free",
       subtitle = "By Region, 1995-2020",
       x = "",
       y = "% of countries",
       fill = "",
       caption = "Source: Freedom Index, Freedom House via Arthur Cheib") +
  scale_fill_brewer(palette = "OrRd", labels = c("Free", "Partially Free", "Not Free")) +
  theme(text = element_text(family = "Cambria", size = 9),
        panel.spacing.x = unit(2, "lines"),
        plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5),
        legend.position = "right",
        strip.background = element_rect(fill = "white"))
ggsave("world.png", width = 9, height = 3, units = "in", dpi = 300)

# default_fill <- brewer.pal(8, "Dark2")[1]
#   
# theme_jw <- function() {
#   theme_linedraw() %+replace%
#     theme(
#       panel.grid.major = element_line(color = "gray", size = 0.05),
#       panel.grid.minor = element_line(color = "gray", size = 0.05),
#       panel.grid.major.x = element_blank(),
#       panel.grid.minor.x = element_blank(),
#       axis.ticks = element_blank(),
#       axis.title.y = element_text(angle = 90, margin = margin(r = 10)),
#       panel.border = element_blank(),
#       plot.title = element_text(hjust = 0.5, face = "bold", margin = margin(b = 10)),
#     )
# }
# 
# ggplot(n_countries) +
#   geom_col(aes(x = Region_Name, y = n_countries), fill = default_fill) +
#   scale_x_discrete() +
#   scale_y_continuous(expand = expansion(0)) +
#   theme_jw() +
#   labs(x = "Region",
#        y = "Number of countries",
#        title = "Number of Countries, by Region")

