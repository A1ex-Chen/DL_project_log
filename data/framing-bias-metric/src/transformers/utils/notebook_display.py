def display(self):
    self.html_code = html_progress_bar(self.value, self.total, self.prefix,
        self.label, self.width)
    if self.inner_table is not None:
        self.html_code += text_to_html_table(self.inner_table)
    if self.child_bar is not None:
        self.html_code += self.child_bar.html_code
    if self.output is None:
        self.output = disp.display(disp.HTML(self.html_code), display_id=True)
    else:
        self.output.update(disp.HTML(self.html_code))
