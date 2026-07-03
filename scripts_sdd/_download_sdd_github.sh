#!/bin/bash
# 下载 flclain/StanfordDroneDataset GitHub 仓库 zip 并检查结构
cd /root
echo "=== 1. 下载 GitHub 仓库 zip (codeload) ==="
for branch in master main; do
  URL="https://codeload.github.com/flclain/StanfordDroneDataset/zip/refs/heads/${branch}"
  echo "  try branch=${branch}: ${URL}"
  wget -q --timeout=90 "${URL}" -O sdd_repo.zip && echo "  OK ${branch}" && break
  echo "  fail ${branch}"
done
ls -la sdd_repo.zip 2>/dev/null || { echo "DOWNLOAD FAILED"; exit 1; }

echo "=== 2. 解压 ==="
rm -rf sdd_repo_extracted
unzip -q sdd_repo.zip -d sdd_repo_extracted 2>&1 | tail -3
echo "解压后顶层目录:"
ls sdd_repo_extracted/

echo "=== 3. 检查结构 (前30个目录) ==="
find sdd_repo_extracted -maxdepth 4 -type d | head -30
echo "---"
echo "annotations.txt 文件示例 (前10个):"
find sdd_repo_extracted -name "annotations.txt" | head -10
echo "---"
echo "总 annotations.txt 数:"
find sdd_repo_extracted -name "annotations.txt" | wc -l

echo "=== 4. 抽查一个文件内容 (前5行) ==="
FIRST=$(find sdd_repo_extracted -name "annotations.txt" | head -1)
if [ -n "${FIRST}" ]; then
  echo "  file: ${FIRST}"
  head -5 "${FIRST}"
  echo "  ..."
  echo "  行数: $(wc -l < "${FIRST}")"
fi

echo "=== 5. 检查 bookstore 场景 ==="
find sdd_repo_extracted -path "*bookstore*" -name "annotations.txt" | head -10
echo "bookstore annotations.txt 数:"
find sdd_repo_extracted -path "*bookstore*" -name "annotations.txt" | wc -l
