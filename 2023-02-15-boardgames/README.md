# Tidy Tuesday - Predicting boardgame ratings (Part 1)


In January 2022, the R Tidy Tuesday featured a dataset of boardgame ratings from BoardGameGeek.com. This notebook (combined with Part 2, to come) investigates  how easily a boardgame's (average) rating can be predicted using data on the characteristics of the game, such as player count, playing time, mechanics involved, the game's designer and the game's publisher.


For this exercise, let's split the dataset into a training set, consisting of 80% of the entries, and a test set, consisting of 20%. The plots that follow will all be based on the training data.

Before we start modeling (in part 2), let's have a quick run-through of the variables available in the dataset.

### The target variable

We will attempt to predict for each boardgame the variable entitled "average", which represents the average of BGG users' ratings for the boardgame, on a scale of 1 to 10.

The plot below shows the distribution of ratings in the training data.



    
![png](output_6_0.png)
    


### Other variables

The dataset comes in two files, *ratings.csv* and *details.csv*. In both files, a row corresponds to a single board game, so the two files can be easily merged. The table below summarizes the number of observations in the training and test sets.






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



Next, let's plot average rating as a function of various potential features (`yearpublished`, `minplayers`, `maxplayers`, `playingtime`, and designer and publisher variables).

### Number of players

We will treat the player-number features as categorical, since they are small and integer-valued.

The plot below shows the distribution of games by minimum player count, as well as average ratings by minimum player count. A minimum player count of 1 seems to be associated with higher average ratings than a minimum player count of 2: presumably players value the ability to play a game in "solo" mode.


    
![png](output_11_0.png)
    


The plot below shows the distribution of games by *maximum* player count, as well as average ratings by maximum player count. Unsurprisingly, even max player counts are more common than odd max player counts. However, there seems to be a small premium associated with even max player counts: games with a max player count of 5 and 7 haveslightly higher average ratings than games with max player counts of 4 and 6, respectively. It is unclear why this should be the case but one possible explanation is that games max player counts of 5 and 7 are associated with more thoughtfulness in the game's design.



    
![png](output_13_0.png)
    


### Playing time

Next, let's look at the playing time for each game.

The plot below shows the distribution of playing times, as well as the average rating by playing time. The bins below *include* their right-most edges, e.g., the "0-30 mins" bin includes games with a playing time of 30 minutes.

The plot shows that longer games have higher average ratings, on average. This is presumably not a *direct* causal effect (i.e., doubling a game's play time does not make it better automatically). Rather, it is likely a combination of two things: (i) a longer play time allows a game to be more complex and more complex games are better on average (at least up to a point); and (ii) in a kind of selection effect, longer games are less accessible (on average) and therefore only *high quality* long games end up getting published.



![png](output_15_0.png)
    


### Year published

The plot below shows the distribution of boardgames by year of publication (binned, somewhat arbitrarily, into 5- and 10-year periods) as well as the average rating by bin.

As the plot shows, more recent games tend to have higher ratings. This pattern should not necessarily be interpreted to mean that modern games are better in any objective sense than older games. That may be true, but the same pattern can also arise if  modern games simply appeal more to today's reviewers, since the rating data is (approximately) a snapshot of today's reviewers' preferences. The pattern may also reflect a preference for newness: for example, it is possible that reviewers give higher scores to games that are new *to them* (and which they are therefore excited to play), which translates into newer games getting higher scores on average.

Incidentally, the plot also illustrates the explosion in boardgame publishing over the past 20 years or so. Over twice as many games were published in the years 2015-19 as were published in the entire 1990s.


    
![png](output_17_0.png)
    


### Mechanics, Designers, and Publishers

The dataset includes data on the game mechanics (e.g., "Dice rolling," "Auctions/Bidding") used in each boardgame. Each game can be associated with multiple mechanics, so values in the "mechanics" field in *details.csv* are comma-separated lists of mechanics, which we have to preprocess before plotting. There are 149 different unique mechanics in the dataset.

What constitutes a unique "mechanic" (and for any given mechanic, whether a board game uses that mechanic or not) is not entirely objective. The mechanics values in the dataset are populated (at least in part) by users of the site; I take the data as is.

The plot below shows the average rating deviation for the 30 most commonly seen mechanics (ranked in descending order of average rating deviation). The plot shows that board games with certain mechanics have higher average ratings than the overall average rating, while board games with other mechanics have lower average ratings.

It is interesting to see the mechanics associated with higher and lower average ratings. Boardgame fans will recognize mechanics used by many popular modern games (e.g., worker placement, cooperative play) among the set of mechanics with positive rating deviations. Among the negative deviation mechanics are mechanics associated with older, less sophisticated games: for example, "Roll / Spin and Move" is a staple mechanic of classic boardgames such as Monopoly, Snakes (or Chutes) and Ladders, and The Game of Life, but is relatively less prevalent in modern games.


    
![png](output_22_0.png)
    


### Designers

We will treat designers as we did mechanics: a board game may have multiple designers, so we split out the "designer" variable, and create dummy variables indicating whether a board game was designed by a particular designer or not. Because there are so many distinct designers in the dataset, we only include designers with at least 10 design credits.

The plot below shows the average rating deviation, by designer, for the twenty designers with the largest (positive) average rating deviations. All of these designers have designed (or co-designed) at least 10 games in the dataset and their games are more highly rated than board games overall. For example, [Hermann Luttman](https://boardgamegeek.com/boardgamedesigner/36105/hermann-luttmann/linkeditems/boardgamedesigner?pageid=1), designer of 33 games, has a postive average rating difference of almost 1.4 (out of 10).



    
![png](output_26_0.png)
    


### Publishers

With the "publisher" variable, we will take a different approach.

Using all the data we have on a boardgame's publisher in our modeling step risks contaminating the predictive analysis. This is because the exercise we have in mind is to predict a boardgame's average rating based on *intrinsic characteristics* of the game, and thus we want to avoid using as predictor any variables that are not intrinsic to the game. Because successful games are likely to be published multiple times (via revised editions and other-language editions), the set of publishers associated with a boardgame is not intrinsic to the game and instead depends, at least in part, on how successful a boardgame has been.

Ideally, we would want to use data on a boardgame's *first* publisher only, since the exercise we have in mind would allow us to observe the game's *first* publisher before making our rating prediction. However, the *boardgamepublisher* variable includes only a list of publishers and does not provide any way (as far as I can see) to identify a game's first publisher. To approximate using the first publisher, we will use data on a single *randomly chosen* publisher per game. Because there are far too many publishers in the dataset to deal with, for modeling and visualization, we will only retain publishers associated with at least 10 boardgames.

    
![png](output_32_0.png)
    


### Conclusion

Based on our visualizations, it seems that at least some of these features will help us predict average ratings. We will continue with the modeling in the Part 2.
