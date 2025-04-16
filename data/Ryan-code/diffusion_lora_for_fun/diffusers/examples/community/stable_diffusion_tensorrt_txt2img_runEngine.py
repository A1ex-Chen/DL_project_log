def runEngine(engine, feed_dict, stream):
    return engine.infer(feed_dict, stream)
