def forward(self, features, targets=None):
    logger = logging.getLogger(__name__)
    logger.warning(
        'Calling forward() may cause unpredicted behavior of PixelDecoder module.'
        )
    return self.forward_features(features)
