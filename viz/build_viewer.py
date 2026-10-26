"""Build viz/episode-viewer.html: a tick-by-tick replay of the experiment logs.

Embeds every src/data/<model>/<order>/experiment_*.json, its llm_calls_*.jsonl
when present, and all recipes and reference trajectories, so the page opens
straight from disk. Other logs can be loaded from the page with "Abrir logs...".

    python viz/build_viewer.py                      # all logs under src/data
    python viz/build_viewer.py --data src/data/gpt-4o/boiled_egg
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")

HEAD = (
    '<!doctype html>\n<html lang="pt-BR">\n<meta charset="utf-8">\n'
    '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
)


def slim_log(path):
    d = json.load(open(path))
    ticks = []
    for c in d["content"]:
        sd = c["statistical_data"]
        cc = c["content"]
        ticks.append({
            "t": c["timestamp"],
            "map": c["map"],
            "actions": c["actions"],
            "queue": cc["action_list"],
            "obs": cc["observation"],
            "calls": cc["content"],
            "comm": [x.get("turn", []) if x.get("call") else [] for x in sd["communication"]],
            "errors": [
                x["validator_error"]["error_message"] + x["format_error"]["error_message"]
                for x in sd["error"]
            ],
            "score": sd["score"],
        })
    # raw LLM input/output written by main.py next to the experiment log (newer runs only)
    raw_path = os.path.join(
        os.path.dirname(path), "llm_calls_" + os.path.basename(path)[len("experiment_"):-len(".json")] + ".jsonl"
    )
    llm = None
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            llm = [json.loads(line) for line in f if line.strip()]
    return {
        "llm": llm,
        "llmFile": os.path.basename(raw_path) if llm is not None else None,
        "file": os.path.basename(path),
        "model": os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(path)))),
        "order": d["content"][0]["order_list"][0] if d["content"] else "",
        "total_score": d["total_score"],
        "finished": d["total_order_finished"],
        "done": d["total_action_list"],
        "ticks": ticks,
    }


def load_recipes():
    out = {}
    for f in sorted(glob.glob(os.path.join(SRC, "prompts/recipe/*.txt"))):
        stem = os.path.basename(f)[:-4]
        level, name = stem.split("_", 1)
        ref_path = os.path.join(SRC, "prompts/reference", stem + "_ref.txt")
        ref = None
        if os.path.exists(ref_path):
            try:
                ref = json.load(open(ref_path))
            except ValueError:
                ref = None
        out[name] = {"level": int(level), "text": open(f).read().strip(), "ref": ref}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.path.join(SRC, "data"), help="folder searched recursively for experiment_*.json")
    parser.add_argument("--out", default=os.path.join(HERE, "episode-viewer.html"))
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.data, "**", "experiment_*.json"), recursive=True))
    logs = [slim_log(p) for p in paths]
    data = {"episodes": logs, "recipes": load_recipes()}
    template = open(os.path.join(HERE, "viewer_template.html")).read()
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    with open(args.out, "w") as f:
        f.write(HEAD + template.replace("/*__DATA__*/null", payload))
    print(f"{len(logs)} episodes -> {os.path.relpath(args.out)} ({len(payload) // 1024} KB)")


if __name__ == "__main__":
    main()
