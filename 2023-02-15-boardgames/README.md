# Tidy Tuesday - Predicting boardgame ratings (Part 1)


In January 2022, the R Tidy Tuesday featured a dataset of boardgame ratings from BoardGameGeek.com. One of the fields in this dataset is the average rating of each boardgame, which is the average of the ratings (on a scale of 1 to 10) given by users of BoardGameGeek.com to the game in question. This notebook (combined with Part 2, to come) investigates whether a boardgame's average rating can be predicted using data on the characteristics of the game, such as the game's player count and playing time, the game mechanics involved in the game, the game's designer(s) and the game's publisher(s).







For this exercise, we first randomly split the dataset into a training set, consisting of 80% of the boardgame entries, and a test set, consisting of the remaining 20%. The plots that follow will all be based on the training data.




Before we start modeling (in part 2), let's have a quick run-through of the variables available in the dataset.

### The target variable

We will attempt to predict for each boardgame the variable entitled "average", which represents the average of BoardGameGeek.com users' ratings for the boardgame, on a scale of 1 to 10.

The plot below shows the distribution of ratings in the training data. As we see from the plot, the modal average boardgame rating is somewhere between 6 and 7, and a substantial majority of average ratings lie between 4 and 8.





    
![png](images/output_6_0.png)
    


### Other variables

The dataset comes in two files, *ratings.csv* and *details.csv*. In both files, a row corresponds to a single boardgame, so the two files can easily be merged. The table below summarizes the number of observations in the training and test sets.







<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Table</th>
      <th>Number of observations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Training (ratings)</td>
      <td>17464</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Training (details)</td>
      <td>17311</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Test (ratings)</td>
      <td>4367</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Test (details)</td>
      <td>4320</td>
    </tr>
  </tbody>
</table>
</div>






Next, let's plot average rating as a function of various potential features (`yearpublished`, `minplayers`, `maxplayers`, `playingtime`, and designer, mechanic, and publisher variables).

### Number of players

We will treat the player-number features (minimum player count and maximum player count) as categorical, rather than numerical, since they are typically small and integer-valued.

The plot below shows the distribution of games by minimum player count (top panel), along with the distribution of average ratings by minimum player count (bottom panel). A minimum player count of 1 seems to be associated with higher average ratings than a minimum player count of 2: presumably players value the ability to play a game in "solo" mode.





    
![png](images/output_11_0.png)
    


The plot below shows the distribution of games by *maximum* player count (top panel), along with average ratings by maximum player count (bottom panel). Unsurprisingly, even-valued maximum player counts are more common than odd-valued max player counts. However, there seems to be a small premium associated with even max player counts: games with maximum player counts of 5 and 7 have slightly higher average ratings than games with maximum player counts of 4 and 6, respectively. It is unclear to me why this should be the case, but one possible explanation is that odd-valued maximum player counts, by virtue of being more unusual, are associated with more thoughtful game design, which results in higher average ratings.





    
![png](images/output_13_0.png)
    


### Playing time

Next, let's look at the playing time for each game.

The plot below shows the distribution of playing times, as well as the average rating by playing time. (Note that the bins below include their maximum values, e.g., the "0-30 mins" bin includes games with a reported playing time of 30 minutes exactly.)

The plot shows that that longer games have higher average ratings, on average. This is presumably not a direct causal effect, i.e., it is not the case that doubling a game's playing time will automatically make the game better. Rather, it is more likely, in my view, to reflect some combination of the following factors: (i) a longer playing time allows a game to be more complex and more complex games (at least up to a point) are better on average; and (ii) longer games are less accessible (on average) and therefore have a higher quality threshold they must exceed in order to get published, which results in an apparent association between longer games and higher quality.






    
![png](images/output_15_0.png)
    


### Year published

The plot below shows the distribution of boardgames by year of publication binned into 5-year periods (top panel), as well as the average rating by bin (bottom panel).

As the plot shows, more recent games tend to have higher ratings. This pattern should not necessarily be interpreted to mean that modern games are better in an objective sense than older games. That may be true, but the same pattern can also arise if modern games simply appeal more to *today's* reviewers, since the rating data reflects today's reviewers' preferences. The pattern may also reflect a preference for newness: if reviewers give higher scores to games that are new *to them*, for example, then more recently released games are likely to have higher ratings on average.

Incidentally, the plot also illustrates the explosion in boardgame publishing over the past 20 years or so. Over twice as many games were published in the years 2015-19 as were published in the entire 1990s.






    
![png](images/output_17_0.png)
    


### Mechanics, Designers, and Publishers

The dataset includes data on the game mechanics (e.g., "Dice rolling," "Auctions/Bidding") used in each boardgame. Each game can be associated with multiple mechanics, so values in the "mechanics" field in *details.csv* are comma-separated lists of mechanics, which we have to preprocess before plotting. There are 149 different unique mechanics in the dataset.

What constitutes a unique mechanic (and for any given mechanic, whether a board game uses that mechanic or not) is not entirely objective. The mechanics values in the dataset are populated by users of BoardGameGeek.com.

The plot below shows the average rating deviation for the 30 most commonly seen mechanics (ranked in descending order of average rating deviation), where a mechanic's "average rating deviation" is the difference between the mean average rating for games featuring that mechanic and the overall mean average rating. The plot shows that board games with certain mechanics have higher average ratings than the overall average rating, while board games with other mechanics have lower average ratings.

It is interesting to see the mechanics associated with higher and lower average ratings. Boardgame fans will recognize mechanics used by many popular modern games (e.g., worker placement, cooperative play) among the set of mechanics with positive rating deviations. Among the negative deviation mechanics are mechanics associated with older games: for example, "Roll / Spin and Move" is a staple mechanic of classic boardgames such as Monopoly, Snakes and Ladders, and The Game of Life, but is relatively less prevalent in modern games.








    
![png](images/output_22_0.png)
    


### Designers

We will treat designers as we did mechanics: a board game may have multiple designers, so we split out the "designer" variable, and create dummy variables indicating whether a board game was designed by a particular designer or not (whether by that designer alone or in conjunction with others). Because there are so many distinct designers in the dataset, we only include designers with at least 10 design credits.

The plot below shows the average rating deviation, by designer, for the twenty designers with the largest positive average rating deviations (among designers with at least 10 credits). For example, [Hermann Luttman](https://boardgamegeek.com/boardgamedesigner/36105/hermann-luttmann/linkeditems/boardgamedesigner?pageid=1), designer of 33 games, has a positive average rating deviation of almost 1.4 (out of 10).




    
![png](images/output_26_0.png)
    


### Publishers

With the "publisher" variable, we will take a different approach.

Using all the data we have on a boardgame's publisher in our modeling step risks contaminating the predictive analysis. This is because we are interested in predicting a boardgame's average rating based on intrinsic characteristics of the game and, unfortunately for us, the set of publishers associated with a game is not an intrinsic characteristic of the game. Rather, the set of publishers associated with a game is *endogenous* to the game's quality and eventual success in the marketplace, since more successful games are more likely to be published multiple times (via, for example, revised editions and other-language editions) and thus are likely to have a greater number of associated publishers.

One possible approach to deal with this endogeneity problem would be to use data on a boardgame's *first* publisher only, since the first publisher associated with a board game is far less likely to be influenced by a board game's eventual success than subsequent publishers. However, while the *boardgamepublisher* variable provides a list of publishers, it does not provide an apparent way to identify a game's first publisher. So instead, we randomly select one publisher per game for use in our analysis. This addresses the concern that the *number* of publishers associated with a game is endogenous to its quality, but it does not entirely eliminate the endogeneity concern.

Because there are far too many publishers in the dataset to deal with, for modeling and visualization, we will only retain publishers associated with at least 10 boardgames.

















    
![png](images/output_32_0.png)
    


### Conclusion

Based on our visualizations, it seems that at least some of these features will help us predict average ratings. We will continue with the modeling in the Part 2.
