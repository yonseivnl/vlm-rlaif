import argparse
import tqdm
import os
import json


def load_json(fpath):
    with open(fpath, "r") as f:
        return json.load(f)


def save_json(data, fpath):
    with open(fpath, "w") as f:
        json.dump(data, f)


parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir")
parser.add_argument("--infer_fname", default="infer_all.json")
args = parser.parse_args()

save_new=True
print(args.pred_dir)
# for possible_out_name in ["infer_all.json", "msvd_qa_infer_all.json", "anet_qa_infer_all.json", "generic.json"]:
for possible_out_name in [args.infer_fname]:
    if os.path.exists(os.path.join(args.pred_dir, possible_out_name)):
        print("Already exists", os.path.join(args.pred_dir, possible_out_name))
        save_new=False
    
if save_new:
    all_preds = []
    for fname in os.listdir(args.pred_dir):
        if fname.endswith(".json"):
            if 'gpt' in fname: continue
            all_preds += load_json(os.path.join(args.pred_dir, fname))
    save_json(all_preds, os.path.join(args.pred_dir, args.infer_fname))
    print("Done")