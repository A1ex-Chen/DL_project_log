def __init__(self, save_dir='marian_converted'):
    assert Path(DEFAULT_REPO).exists(
        ), 'need git clone git@github.com:Helsinki-NLP/Tatoeba-Challenge.git'
    reg = self.make_tatoeba_registry()
    self.download_metadata()
    self.registry = reg
    reg_df = pd.DataFrame(reg, columns=['id', 'prepro', 'url_model',
        'url_test_set'])
    assert reg_df.id.value_counts().max() == 1
    reg_df = reg_df.set_index('id')
    reg_df['src'] = reg_df.reset_index().id.apply(lambda x: x.split('-')[0]
        ).values
    reg_df['tgt'] = reg_df.reset_index().id.apply(lambda x: x.split('-')[1]
        ).values
    released_cols = ['url_base', 'pair', 'short_pair', 'chrF2_score',
        'bleu', 'brevity_penalty', 'ref_len', 'src_name', 'tgt_name']
    released = pd.read_csv('Tatoeba-Challenge/models/released-models.txt',
        sep='\t', header=None).iloc[:-1]
    released.columns = released_cols
    released['fname'] = released['url_base'].apply(lambda x: remove_suffix(
        remove_prefix(x,
        'https://object.pouta.csc.fi/Tatoeba-Challenge/opus'), '.zip'))
    released['2m'] = released.fname.str.startswith('2m')
    released['date'] = pd.to_datetime(released['fname'].apply(lambda x:
        remove_prefix(remove_prefix(x, '2m-'), '-')))
    released['base_ext'] = released.url_base.apply(lambda x: Path(x).name)
    reg_df['base_ext'] = reg_df.url_model.apply(lambda x: Path(x).name)
    metadata_new = reg_df.reset_index().merge(released.rename(columns={
        'pair': 'id'}), on=['base_ext', 'id'])
    metadata_renamer = {'src': 'src_alpha3', 'tgt': 'tgt_alpha3', 'id':
        'long_pair', 'date': 'train_date'}
    metadata_new = metadata_new.rename(columns=metadata_renamer)
    metadata_new['src_alpha2'] = metadata_new.short_pair.apply(lambda x: x.
        split('-')[0])
    metadata_new['tgt_alpha2'] = metadata_new.short_pair.apply(lambda x: x.
        split('-')[1])
    DROP_COLS_BOTH = ['url_base', 'base_ext', 'fname']
    metadata_new = metadata_new.drop(DROP_COLS_BOTH, 1)
    metadata_new['prefer_old'] = metadata_new.long_pair.isin([])
    self.metadata = metadata_new
    assert self.metadata.short_pair.value_counts().max(
        ) == 1, 'Multiple metadata entries for a short pair'
    self.metadata = self.metadata.set_index('short_pair')
    mapper = pd.read_csv(LANG_CODE_PATH)
    mapper.columns = ['a3', 'a2', 'ref']
    self.iso_table = pd.read_csv(ISO_PATH, sep='\t').rename(columns=lambda
        x: x.lower())
    more_3_to_2 = self.iso_table.set_index('id').part1.dropna().to_dict()
    more_3_to_2.update(mapper.set_index('a3').a2.to_dict())
    self.alpha3_to_alpha2 = more_3_to_2
    self.model_card_dir = Path(save_dir)
    self.constituents = GROUP_MEMBERS
