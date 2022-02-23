library(tidyverse)
library(lubridate)
library(ggplot2)

rm(list = ls())

airmen <- read_csv("airmen.csv")

ggplot(airmen) + geom_histogram(aes(x = graduation_date))

date_from_aerial_victory <- function(x) {
  str_extract(x, regex("(January|February|March|April|May|June|July|August|September|October|November|December) [:digit:]+, [:digit:]+")) %>% 
    mdy()
}

airmen_vic <- airmen %>%
  mutate() %>%
  separate(
    aerial_victory_credits,
    into = paste0("credit", 1:4),
    sep = ";") %>%
  mutate(date_credit1 = date_from_aerial_victory(credit1),
         date_credit2 = date_from_aerial_victory(credit2),
         date_credit3 = date_from_aerial_victory(credit3),
         date_credit4 = date_from_aerial_victory(credit4))

victories <- airmen_vic %>%
  pivot_longer(cols = c("date_credit1", "date_credit2", "date_credit3", "date_credit4"),
               names_to = "credit",
               names_prefix = "date_credit",
               values_to = "credit_date") %>%
  filter(!is.na(credit_date))

distinguished_99 <- tribble(
  ~date, ~description,
  ymd("1943-07-01"), "99th Fighter Squadron receives\nDistinguished Unit Citation,\nSicily, Italy,\nJun-Jul 1943",
  ymd("1944-05-14"), "99th Fighter Squadron receives\nDistinguished Unit Citation,\nCassino, Italy,\n12-14 May 1944",
  ymd("1945-03-24"), "99th Fighter Squadron receives\nDistinguished Unit Citation, Germany,\n24 Mar 1945"
)

dev.new(height = 6, width = 10, noRStudioGD = TRUE)
ggplot(distinguished_99) + 
  geom_text(aes(x = date+3, label = description, y = 19.9), hjust = -0.0, vjust = 1.0) +
  coord_cartesian(clip = "off") +
  scale_y_continuous(expand = expansion(0)) + 
  scale_x_date(breaks = c(ymd("1943-01-01")
                          , ymd("1944-01-01")
                          , ymd("1945-01-1")
                          , ymd("1946-01-01")
                          , ymd("1943-07-01")
                          , ymd("1944-07-01")
                          , ymd("1945-07-01")),
               date_labels = "%b %Y",
               limits = c(ymd("1943-01-01"),
                          ymd("1946-01-01"))) +
  geom_segment(aes(x = date, xend = date, y = 0.00, yend = 20.00), linetype = "dotted") +
  geom_histogram(data = victories, aes(x = credit_date), binwidth = 1) +
  theme_minimal() + 
  theme(plot.title = element_text(hjust = 0.5, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, face = "bold"),
        axis.ticks = element_line(color = "black"),
        axis.line = element_line(color = "black"),
        panel.grid = element_blank()) +
  labs(x = "",
       y = "",
       title = "Aerial Victory Credits, Tuskegee Airmen",
       subtitle = "1943 to 1945",
       caption = "Source: Veterans Advocacy Tableau User Group; Commemorative Air Force; Combat Squadrons of the Air Force, World War II, Edited by Maurer Maurer, 1982.")
ggsave("airmen.png")
