@property
def model_names(self):
    assert len(self.models
        ) > 0, "Please make sure you provide at least one model name / model identifier, *e.g.* `--models bert-base-cased` or `args.models = ['bert-base-cased']."
    return self.models
