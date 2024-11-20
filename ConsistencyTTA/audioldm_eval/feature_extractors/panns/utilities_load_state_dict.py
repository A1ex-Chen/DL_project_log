def load_state_dict(self, resume_iteration):
    self.statistics_dict = pickle.load(open(self.statistics_path, 'rb'))
    resume_statistics_dict = {'bal': [], 'test': []}
    for key in self.statistics_dict.keys():
        for statistics in self.statistics_dict[key]:
            if statistics['iteration'] <= resume_iteration:
                resume_statistics_dict[key].append(statistics)
    self.statistics_dict = resume_statistics_dict
