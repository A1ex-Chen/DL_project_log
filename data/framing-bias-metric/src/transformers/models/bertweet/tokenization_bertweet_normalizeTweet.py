def normalizeTweet(self, tweet):
    """
        Normalize a raw Tweet
        """
    for punct in self.special_puncts:
        tweet = tweet.replace(punct, self.special_puncts[punct])
    tokens = self.tweetPreprocessor.tokenize(tweet)
    normTweet = ' '.join([self.normalizeToken(token) for token in tokens])
    normTweet = normTweet.replace('cannot ', 'can not ').replace("n't ",
        " n't ").replace("n 't ", " n't ").replace("ca n't", "can't").replace(
        "ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're "
        ).replace("'s ", " 's ").replace("'ll ", " 'll ").replace("'d ", " 'd "
        ).replace("'ve ", " 've ")
    normTweet = normTweet.replace(' p . m .', '  p.m.').replace(' p . m ',
        ' p.m ').replace(' a . m .', ' a.m.').replace(' a . m ', ' a.m ')
    return ' '.join(normTweet.split())
