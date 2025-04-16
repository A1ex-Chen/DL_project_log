def dataframe(self):
    """Get a dataframe of all task accuracies"""
    avg_accuracy = {k: v.avgs for k, v in self.meters.items()}
    return pd.DataFrame(avg_accuracy)
