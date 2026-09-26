"""Count all-reduce re-clones in a generated wrapper and report how many are dead.

A ``_numa_ar_<buf>_n<k>`` name that appears exactly once in the file was
defined and never read: a full-tensor per-socket clone nothing consumes.
"""
import sys, os, re, glob, collections

def scan(path):
    names = collections.Counter()
    clone_defs = 0
    for line in open(path, errors="ignore"):
        if "_numa_ar_" in line:
            for m in re.findall(r"_numa_ar_[A-Za-z0-9_]+", line):
                names[m] += 1
            if re.match(r"\s*_numa_ar_[A-Za-z0-9_]+ = torch\.ops\.aten\.clone", line):
                clone_defs += 1
    dead = [n for n, c in names.items() if c == 1]
    return len(names), len(dead), clone_defs, sorted(dead)[:6]

roots = sys.argv[1:]
files = []
for r in roots:
    if os.path.isfile(r):
        files.append(r)
    else:
        for f in glob.glob(os.path.join(r, "**", "*.py"), recursive=True):
            if os.path.getsize(f) > 40000:
                files.append(f)
best = None
for f in files:
    tot, dead, clones, sample = scan(f)
    if tot and (best is None or tot > best[1]):
        best = (f, tot, dead, clones, sample)
if best is None:
    print("no _numa_ar_ names found")
else:
    f, tot, dead, clones, sample = best
    print("file=%s" % f)
    print("numa_ar_names=%d  dead(referenced once)=%d  clone_defs=%d" % (tot, dead, clones))
    if sample:
        print("dead sample: %s" % ", ".join(sample))
