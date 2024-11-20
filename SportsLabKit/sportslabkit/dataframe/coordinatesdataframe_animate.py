def animate(i):
    """Function to animate the data.

            Each frame it sets the data for the players and the ball.
            """
    ball.set_data(df_ball.loc[:, (slice(None), 'x')].iloc[i], df_ball.loc[:,
        (slice(None), 'y')].iloc[i])
    away.set_data(df_away.loc[:, (slice(None), 'x')].iloc[i], df_away.loc[:,
        (slice(None), 'y')].iloc[i])
    home.set_data(df_home.loc[:, (slice(None), 'x')].iloc[i], df_home.loc[:,
        (slice(None), 'y')].iloc[i])
    return ball, away, home
