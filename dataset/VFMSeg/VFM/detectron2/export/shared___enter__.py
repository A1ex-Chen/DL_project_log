def __enter__(self):
    self.org_ws = workspace.CurrentWorkspace()
    if self.ws_name is not None:
        workspace.SwitchWorkspace(self.ws_name, True)
    if self.is_reset:
        workspace.ResetWorkspace()
    return workspace
