def visualize_frames(self, save_path: PathLike, ball_key: str='ball',
    home_key: str='0', away_key: str='1', marker_kwargs: (dict[str, Any] |
    None)=None, ball_kwargs: (dict[str, Any] | None)=None, home_kwargs: (
    dict[str, Any] | None)=None, away_kwargs: (dict[str, Any] | None)=None,
    save_kwargs: (dict[str, Any] | None)=None):
    """Visualize multiple frames using matplotlib.animation.FuncAnimation.

        Visualizes the frames and generates a pitch animation. The `CoordinatesDataFrame` is expected to already have been normalized so that the pitch is 105x68, e.g. coordinates on the x-axis range from 0 to 105 and coordinates on the y-axis range from 0 to 68.

        To customize the animation, you can pass keyword arguments to `matplotlib.animation.FuncAnimation`. For example, to change the frame rate, you can pass `fps=30` to `save_kwargs` by, e.g. `codf.visualize_frames("animation.gif", save_kwargs={"fps": 30})`. See the `matplotlib.animation.FuncAnimation` documentation for more information.

        Similarly, you can pass keyword arguments to change the appearance of the markers. For example, to change the size of the markers, you can pass `ms=6` to `away_kwargs` by, e.g. `codf.visualize_frames("animation.gif", away_kwargs={"ms": 6})`. See the `matplotlib.pyplot.plot` documentation for more information.  Note that `marker_kwargs` will be used for all markers but will be overwritten by `ball_kwargs`, `home_kwargs` and `away_kwargs` if a dictionary with the same key is passed (later dictionaries take precedence).

        Args:
            frame_idx: Frame number.
            save_path: Path to save the image. Defaults to None.
            ball_key: Key (TeamID) for the ball. Defaults to "ball".
            home_key: Key (TeamID) for the home team. Defaults to "0".
            away_key: Key (TeamID) for the away team. Defaults to "1".
            marker_kwargs: Keyword arguments for the markers.
            ball_kwargs: Keyword arguments specifically for the ball marker.
            home_kwargs: Keyword arguments specifically for the home team markers.
            away_kwargs: Keyword arguments specifically for the away team markers.
            save_kwargs: Keyword arguments for the save function.

        Note:
            `marker_kwargs` will be used for all markers but will be overwritten by `ball_kwargs`, `home_kwargs` and `away_kwargs`. All keyword arguments are passed to `plt.plot`. `save_kwargs` are passed to `FuncAnimation.save`.

        Warning:
            All keyword arguments are passed either to `plt.plot` and `FuncAnimation.save`. If you pass an invalid keyword argument, you will get an error.

        Example:
            >>> codf = load_codf("/path/to/codf.csv")
            >>> codf.visualize_frames("/path/to/save.mp4")
            ...
            # Heres a demo using random data
            >>> codf = CoordinatesDataFrame.from_numpy(np.random.randint(0, 50, (1, 23, 2)))
            >>> codf = codf.loc[codf.index.repeat(5)] # repeat the same frame 5 times
            >>> codf += np.array([[0,1,2,3,4]]).T # add some movment
            >>> codf.visualize_frames('visualize_frames.gif', save_kwargs={'fps':2})

        .. image:: /_static/visualize_frames.gif
        """
    _marker_kwargs = merge_dicts({'marker': 'o', 'markeredgecolor': 'None',
        'linestyle': 'None'}, marker_kwargs)
    _ball_kwargs = merge_dicts(_marker_kwargs, {'zorder': 3, 'ms': 6,
        'markerfacecolor': 'w'}, marker_kwargs, ball_kwargs)
    _home_kwargs = merge_dicts(_marker_kwargs, {'zorder': 10, 'ms': 10,
        'markerfacecolor': 'b'}, marker_kwargs, home_kwargs)
    _away_kwargs = merge_dicts(_marker_kwargs, {'zorder': 10, 'ms': 10,
        'markerfacecolor': 'r'}, marker_kwargs, away_kwargs)
    _save_kwargs = merge_dicts({'dpi': 100, 'fps': 10, 'savefig_kwargs': {
        'facecolor': 'black', 'pad_inches': 0.0}}, save_kwargs)
    _df = self.copy()
    df_ball = _df[ball_key]
    df_home = _df[home_key]
    df_away = _df[away_key]
    pitch = Pitch(pitch_color='black', line_color=(0.3, 0.3, 0.3),
        pitch_type='custom', pitch_length=105, pitch_width=68, label=False)
    fig, ax = pitch.draw(figsize=(8, 5.2))
    ball, *_ = ax.plot([], [], **_ball_kwargs)
    away, *_ = ax.plot([], [], **_away_kwargs)
    home, *_ = ax.plot([], [], **_home_kwargs)

    def animate(i):
        """Function to animate the data.

            Each frame it sets the data for the players and the ball.
            """
        ball.set_data(df_ball.loc[:, (slice(None), 'x')].iloc[i], df_ball.
            loc[:, (slice(None), 'y')].iloc[i])
        away.set_data(df_away.loc[:, (slice(None), 'x')].iloc[i], df_away.
            loc[:, (slice(None), 'y')].iloc[i])
        home.set_data(df_home.loc[:, (slice(None), 'x')].iloc[i], df_home.
            loc[:, (slice(None), 'y')].iloc[i])
        return ball, away, home
    anim = FuncAnimation(fig, animate, frames=len(_df), blit=True)
    try:
        anim.save(save_path, **_save_kwargs)
    except Exception:
        logger.error(
            'BrokenPipeError: Saving animation failed, which might be an ffmpeg problem. Trying again with different codec.'
            )
        _save_kwargs['extra_args'] = ['-vcodec', 'mpeg4', '-pix_fmt', 'yuv420p'
            ]
        try:
            anim.save(save_path, **_save_kwargs)
        except Exception as e:
            logger.error(
                'Saving animation failed again. Exiting without saving the animation.'
                )
            print(e)
