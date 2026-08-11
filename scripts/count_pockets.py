import os
import lmdb

root = "/project/data/lit_pcba"
total = 0
lines = []
for t in sorted(os.listdir(root)):
    p = os.path.join(root, t, "pockets.lmdb")
    if os.path.isfile(p):
        env = lmdb.open(p, subdir=False, readonly=True, lock=False)
        n = env.stat()["entries"]
        env.close()
        total += n
        lines.append(f"{t}: {n} 个口袋")
    else:
        lines.append(f"{t}: 无 pockets.lmdb")
lines.append(f"TOTAL: {total}")
print("\n".join(lines))
