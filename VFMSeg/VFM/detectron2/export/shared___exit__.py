def __exit__(self, *args):
    if self.is_cleanup:
        workspace.ResetWorkspace()
    if self.ws_name is not None:
        workspace.SwitchWorkspace(self.org_ws)
