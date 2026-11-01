#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=$(mktemp -d)
trap 'rm -f "$build_dir/mapping_test" "$build_dir/gpu_slim_test"; rmdir "$build_dir"' EXIT
compiler=${CXX:-g++}
"$compiler" -std=c++11 -Wall -Wextra -pedantic -I"$repo_root" \
  "$repo_root/tests/null_mapping/mapping_test.cpp" -o "$build_dir/mapping_test"
"$build_dir/mapping_test"
"$compiler" -std=c++11 -Wall -Wextra -pedantic -I"$repo_root/nep_gpu/force" \
  "$repo_root/tests/null_mapping/gpu_slim_test.cpp" -o "$build_dir/gpu_slim_test"
"$build_dir/gpu_slim_test"
