def show_df(df, theme='dark'):
    from IPython.display import display

    def dark(styler):
        styler.applymap(lambda x: 'color: white')
        styler.set_table_styles([{'selector': 'th', 'props': [('color',
            'white'), ('background-color', '#555555')]}])
        styler.apply(lambda x: ['background: #333333' for _ in x], axis=1)
        return styler

    def light(styler):
        raise NotImplementedError('Light theme not implemented yet.')
    style = dark if theme == 'dark' else light
    return display(df.style.pipe(style))
