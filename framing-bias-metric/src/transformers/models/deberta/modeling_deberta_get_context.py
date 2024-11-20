def get_context(self):
    if self.context_stack is not None:
        if self.count >= len(self.context_stack):
            self.context_stack.append(DropoutContext())
        ctx = self.context_stack[self.count]
        ctx.dropout = self.drop_prob
        self.count += 1
        return ctx
    else:
        return self.drop_prob
