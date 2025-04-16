def extract_blocks_into_list(model, blocks):
    for module in model.children():
        if isinstance(module, LinearAddBlock) or isinstance(module,
            RealVGGBlock):
            blocks.append(module)
        else:
            extract_blocks_into_list(module, blocks)
