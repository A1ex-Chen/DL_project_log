def handle_item(fieldarg, content):
    par = nodes.paragraph()
    par += addnodes.literal_strong('', fieldarg)
    if fieldarg in types:
        par += nodes.Text(' (')
        fieldtype = types.pop(fieldarg)
        if len(fieldtype) == 1 and isinstance(fieldtype[0], nodes.Text):
            typename = u''.join(n.astext() for n in fieldtype)
            typename = typename.replace('int', 'python:int')
            typename = typename.replace('long', 'python:long')
            typename = typename.replace('float', 'python:float')
            typename = typename.replace('type', 'python:type')
            par.extend(self.make_xrefs(self.typerolename, domain, typename,
                addnodes.literal_emphasis, **kw))
        else:
            par += fieldtype
        par += nodes.Text(')')
    par += nodes.Text(' -- ')
    par += content
    return par
