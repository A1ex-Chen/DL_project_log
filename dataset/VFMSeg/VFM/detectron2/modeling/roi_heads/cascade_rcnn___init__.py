@configurable
def __init__(self, *, box_in_features: List[str], box_pooler: ROIPooler,
    box_heads: List[nn.Module], box_predictors: List[nn.Module],
    proposal_matchers: List[Matcher], **kwargs):
    """
        NOTE: this interface is experimental.

        Args:
            box_pooler (ROIPooler): pooler that extracts region features from given boxes
            box_heads (list[nn.Module]): box head for each cascade stage
            box_predictors (list[nn.Module]): box predictor for each cascade stage
            proposal_matchers (list[Matcher]): matcher with different IoU thresholds to
                match boxes with ground truth for each stage. The first matcher matches
                RPN proposals with ground truth, the other matchers use boxes predicted
                by the previous stage as proposals and match them with ground truth.
        """
    assert 'proposal_matcher' not in kwargs, "CascadeROIHeads takes 'proposal_matchers=' for each stage instead of one 'proposal_matcher='."
    kwargs['proposal_matcher'] = proposal_matchers[0]
    num_stages = self.num_cascade_stages = len(box_heads)
    box_heads = nn.ModuleList(box_heads)
    box_predictors = nn.ModuleList(box_predictors)
    assert len(box_predictors
        ) == num_stages, f'{len(box_predictors)} != {num_stages}!'
    assert len(proposal_matchers
        ) == num_stages, f'{len(proposal_matchers)} != {num_stages}!'
    super().__init__(box_in_features=box_in_features, box_pooler=box_pooler,
        box_head=box_heads, box_predictor=box_predictors, **kwargs)
    self.proposal_matchers = proposal_matchers
