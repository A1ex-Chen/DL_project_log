def send_analysis_error(self, analysis_error, context):
    message = pm.AnalysisError()
    message.error_message = str(analysis_error)
    if analysis_error.file_context is not None:
        message.file_context.file_path.components.extend(analysis_error.
            file_context.file_path.split(os.sep))
        message.file_context.line_number = (analysis_error.file_context.
            line_number if analysis_error.file_context.line_number is not
            None else 0)
    self._send_message(message, 'analysis_error', context)
