import argparse

from mbchl.utils import flatten_dict, pretty_table, read_json, read_yaml


def main():
    for input_ in args.inputs:
        print(input_)
        if input_.endswith(".json"):
            data = read_json(input_)
        elif input_.endswith(".yaml"):
            data = read_yaml(input_)
        else:
            raise ValueError("Input file must be .json or .yaml")
        data = {k: flatten_dict(v) for k, v in data.items()}
        pretty_table(data, order_by=args.order_by, decimals=args.decimals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--decimals", type=int)
    parser.add_argument("--order_by")
    args = parser.parse_args()
    main()
