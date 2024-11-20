def get_log_dir(self):
    if (self.logger_save_dir is None and self.logger_project is None and 
        self.logger_version is None):
        return os.path.join(self.logger.save_dir, self.logger._project,
            self.logger.version)
    else:
        return os.path.join(self.logger_save_dir, self.logger_project, self
            .logger_version)
