# Electrolyte LAMMPS Compare

一键执行完整验证：

```bash
cd /data/home/wuxingxing/codespace/lammps-MatPL/KKNEP/merge-qnep-opt/test_qnep/electrolyte
./run_all_validation.sh
```

一键清理验证输出：

```bash
cd /data/home/wuxingxing/codespace/lammps-MatPL/KKNEP/merge-qnep-opt/test_qnep/electrolyte
./clean_all_validation.sh --yes
```

NEP5 测试：

```bash
cd electrolyte/lmps-infer
sbatch job_lmps_compare.sh --structure all --kspace both --threshold 1e-4
```

NEP4 测试：

```bash
cd electrolyte/lmps-infer-nep4
sbatch job_lmps_compare.sh --structure all --kspace both --threshold 1e-4
```

两个命令都会对 `part000/part020/part040/part060/part080` 分别执行 `ewald` 和 `pppm` 推理，并在对应目录下生成 `auto_outputs/`、`auto_logs/` 和 `compare_results/`。

MatPL 参考值目录为：

```text
MatPL-infer/ref-result
ref-result-cpu是cpu接口推理接口，与GPU区别是，cpu端采用的是FP64精度，实验中参考值引用的是ref-result,来自GPU的推理，核心计算采用FP32
```
