def dark(styler):
    styler.applymap(lambda x: 'color: white')
    styler.set_table_styles([{'selector': 'th', 'props': [('color', 'white'
        ), ('background-color', '#555555')]}])
    styler.apply(lambda x: ['background: #333333' for _ in x], axis=1)
    return styler
