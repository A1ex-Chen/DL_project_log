def __call__(self, prompts, aggregation=True, **kwargs):
    generations = self.generate_on_prompts(self.generator, prompts, **kwargs)
    if aggregation:
        generations = self.agg_generations(generations)
    generations = list(itertools.chain(*[ints for _, ints in generations.
        items()]))
    generations = [g.strip() for g in generations]
    return generations
