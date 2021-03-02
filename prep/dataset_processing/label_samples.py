import argparse
import json
import os


def main(args: argparse.Namespace):
    with open(args.dataset_description, "r") as df:
        samples = json.load(df)

    new_samples = []
    for sample in samples:
        path = sample["path"]
        string = sample["string"].replace("'", "").replace("-", "")
        # types: text, num, date, alpha_num, plz, spec_char, shape, signature
        if "generated_num" in path or ("rimes" in path and string.isnumeric()):
            sample["type"] = "num"
        elif "generated_text" in path or ("rimes" in path and string.isalpha()):
            sample["type"] = "text"
        elif "generated_alpha_num" in path or ("rimes" in path and string.isalnum()):
            sample["type"] = "alpha_num"
        elif "generated_plz" in path:
            sample["type"] = "plz"
        elif "generated_date" in path:
            sample["type"] = "date"
        elif "generated_spec_char" in path:
            sample["type"] = "spec_char"
        elif "generated_shape" in path:
            sample["type"] = "shape"
        elif "signature" in path:
            sample["type"] = "signature"
        else:
            # TODO: code is not perfect and does not catch all cases but should be sufficient
            print(string, path)
            continue
        new_samples.append(sample)

    out_dir = os.path.dirname(args.dataset_description)
    with open(os.path.join(out_dir, args.out_filename), "w") as outf:
        json.dump(new_samples, outf, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_description", type=str,
                        help="path to the dataset description that contains the unlabelled samples")
    parser.add_argument("out_filename", type=str, help="name of the new dataset description file")
    main(parser.parse_args())