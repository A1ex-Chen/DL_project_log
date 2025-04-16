def on_pick(event):
    index = event.ind[0]
    pick_info = scatter_to_pick_info_data[event.artist][index]
    on_pick_callback(pick_info)
