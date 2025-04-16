def main():
    args = get_parser().parse_args()
    model = get_model(args.model)
    outputs = get_outputs(args.JSONS, args.output_dir, args.suffix, args.
        output_format)
    transform(model, args.JSONS, outputs, args.output_format)
