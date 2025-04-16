def visualize_frame(self, frame_idx: int, save_path: (PathLike | None)=None,
    ball_key: str='ball', home_key: str='0', away_key: str='1',
    marker_kwargs: (dict[str, Any] | None)=None, ball_kwargs: (dict[str,
    Any] | None)=None, home_kwargs: (dict[str, Any] | None)=None,
    away_kwargs: (dict[str, Any] | None)=None, save_kwargs: (dict[str, Any] |
    None)=None):
    """Visualize a single frame.

        Visualize a frame given a frame number and save it to a path. The `CoordinatesDataFrame` is expected to already have been normalized so that the pitch is 105x68, e.g. coordinates on the x-axis range from 0 to 105 and coordinates on the y-axis range from 0 to 68.

        Similarly, you can pass keyword arguments to change the appearance of the markers. For example, to change the size of the markers, you can pass `ms=6` to `away_kwargs` by, e.g. `codf.visualize_frames("animation.gif", away_kwargs={"ms": 6})`. See the `matplotlib.pyplot.plot` documentation for more information. Note that `marker_kwargs` will be used for all markers but will be overwritten by `ball_kwargs`, `home_kwargs` and `away_kwargs` if a dictionary with the same key is passed (later dictionaries take precedence).

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
            `marker_kwargs` will be used for all markers but will be overwritten by `ball_kwargs`, `home_kwargs` and `away_kwargs`. All keyword arguments are passed to `plt.plot`. `save_kwargs` are passed to `plt.savefig`.

        Warning:
            All keyword arguments are passed to `plt.plot`. If you pass an invalid keyword argument, you will get an error.

        Example:
            >>> codf = CoordinatesDataFrame.from_numpy(np.random.randint(0, 105, (1, 23, 2)))
            >>> codf.visualize_frame(0)

        .. image:: /_static/visualize_frame.png
        """
    _marker_kwargs = merge_dicts({'marker': 'o', 'markeredgecolor': 'None',
        'linestyle': 'None'}, marker_kwargs)
    _ball_kwargs = merge_dicts(_marker_kwargs, {'zorder': 3, 'ms': 6,
        'markerfacecolor': 'w'}, marker_kwargs, ball_kwargs)
    _home_kwargs = merge_dicts(_marker_kwargs, {'zorder': 10, 'ms': 10,
        'markerfacecolor': 'b'}, marker_kwargs, home_kwargs)
    _away_kwargs = merge_dicts(_marker_kwargs, {'zorder': 10, 'ms': 10,
        'markerfacecolor': 'r'}, marker_kwargs, away_kwargs)
    _save_kwargs = merge_dicts({'facecolor': 'black', 'pad_inches': 0.0},
        save_kwargs)
    _df = self.copy()
    _df = _df[_df.index == frame_idx]
    df_ball = _df[ball_key]
    df_home = _df[home_key]
    df_away = _df[away_key]
    pitch = Pitch(pitch_color='black', line_color=(0.3, 0.3, 0.3),
        pitch_type='custom', pitch_length=105, pitch_width=68, label=False)
    fig, ax = pitch.draw(figsize=(8, 5.2))
    ax.plot(df_ball.loc[:, (slice(None), 'x')], df_ball.loc[:, (slice(None),
        'y')], **_ball_kwargs)
    ax.plot(df_away.loc[:, (slice(None), 'x')], df_away.loc[:, (slice(None),
        'y')], **_away_kwargs)
    ax.plot(df_home.loc[:, (slice(None), 'x')], df_home.loc[:, (slice(None),
        'y')], **_home_kwargs)
    if save_path is not None:
        fig.savefig(save_path, **_save_kwargs)
