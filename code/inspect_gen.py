import json, os
base = os.path.expanduser("~/mdrp-repro")
g = [json.loads(l) for l in open(f"{base}/generations.jsonl")]
print("total:", len(g))
print("empty:", sum(1 for x in g if not x["response"].strip()))
print("with <think>:", sum(1 for x in g if "<think" in x["response"] or "</think" in x["response"]))
lens = [len(x["response"]) for x in g]
print("resp len: min %d  mean %d  max %d" % (min(lens), sum(lens)//len(lens), max(lens)))
i0 = g[0]["id"]
print("\n--- sample: id %s role %s cued '%s' ---" % (i0, g[0]["role"], g[0]["cued_title"]))
for x in g:
    if x["id"] == i0:
        print("[%-18s] %s" % (x["condition"], x["response"][:90]))
