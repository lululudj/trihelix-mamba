#!/bin/bash
# 创建 N<=8 的数据子集（hardlink，不占额外空间），避免 mamba3 在 N=12 时 OOM
set -e
SRC=/mnt/e/three_chain_v3/data_split
DST=/mnt/e/three_chain_v3/data_split_m3

mkdir -p "$DST"/train "$DST"/val "$DST"/test

for split in train val test; do
    for f in "$SRC"/$split/scen_06_*.npz "$SRC"/$split/scen_08_*.npz; do
        [ -f "$f" ] && ln "$f" "$DST"/$split/ 2>/dev/null || true
    done
done

echo "DONE"
for s in train val test; do
    count=$(ls "$DST"/$s 2>/dev/null | wc -l)
    echo "$s: $count files"
done
