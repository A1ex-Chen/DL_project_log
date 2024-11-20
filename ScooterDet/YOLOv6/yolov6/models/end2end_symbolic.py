@staticmethod
def symbolic(g, boxes, scores, plugin_version='1', shareLocation=1,
    backgroundLabelId=-1, numClasses=80, topK=1000, keepTopK=100,
    scoreThreshold=0.25, iouThreshold=0.45, isNormalized=0, clipBoxes=0,
    scoreBits=16, caffeSemantics=1):
    out = g.op('TRT::BatchedNMSDynamic_TRT', boxes, scores, shareLocation_i
        =shareLocation, plugin_version_s=plugin_version,
        backgroundLabelId_i=backgroundLabelId, numClasses_i=numClasses,
        topK_i=topK, keepTopK_i=keepTopK, scoreThreshold_f=scoreThreshold,
        iouThreshold_f=iouThreshold, isNormalized_i=isNormalized,
        clipBoxes_i=clipBoxes, scoreBits_i=scoreBits, caffeSemantics_i=
        caffeSemantics, outputs=4)
    nums, boxes, scores, classes = out
    return nums, boxes, scores, classes
