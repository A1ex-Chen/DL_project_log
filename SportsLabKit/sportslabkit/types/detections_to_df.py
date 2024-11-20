def to_df(self):
    preds = self.preds
    df = pd.DataFrame(preds, columns=['bbox_left', 'bbox_top', 'bbox_width',
        'bbox_height', 'conf', 'class'])
    return df
