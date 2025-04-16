def find_modules(model, mclass=nn.Conv2d):
    return [i for i, m in enumerate(model.module_list) if isinstance(m, mclass)
        ]
