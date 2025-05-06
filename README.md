
## Lammps-nep-only 编译安装
此版本为 nep 模型的lammps 力场接口，仅支持 MatPL训练的 NEP 模型和 GPUMD 训练的 NEP4，不支持其他模型；力场接口增加了对 GPU 加速的支持，支持跨节点、跨节点GPU；力场接口支持输出多模型偏差。

lammps 源码安装需要您下载 lammps 源码、加载编译器、编译源码，过程如下所示。

### 1. 准备 MatPL 力场接口和 lammps 源码

#### MATPL 力场接口源码
MATPL 力场接口源码位于 [MatPL 源码目录/src/lmps] 下，您也可以通过 github 下载 MATPL 力场接口源码，或者下载 release 包。
- 通过 github 或 gitee clone 源码:
```bash
git clone -b https://github.com/LonxunQuantum/lammps-MatPL/tree/nep-only
或
git clone -b https://gitee.com/pfsuo/lammps-MatPL.git
```

- 或下载release 包:
```bash
wget https://github.com/LonxunQuantum/lammps-MatPL/archive/refs/tags/2025.3.zip
或
wget https://gitee.com/pfsuo/lammps-MatPL/repository/archive/2025.3

unzip 2025.3.zip    #解压源码
```
MatPL 力场接口源码目录如下所示
```
├── Makefile.mpi
├── MATPL/
├── nep_lmps_demo/
└── README.md
```

#### Lammps 源码

lammps 源码请访问 [lammps github 仓库](https://github.com/lammps/lammps/tree/stable#) 下载，这里推荐下载 `stable 版本`。

#### lammps 源码目录下设置力场文件

- 复制 Makefile.mpi 文件到 lammps/src/MAKE/目录下

- 复制 MATPL 目录到 lammps/src/目录下

### 2. 加载编译环境
首先检查 `cuda/11.8`，`intel/2020`，`gcc8.n`是否加载；

- 对于 `intel/2020`编译套件，使用了它的 `ifort` 和 `icc` 编译器(`19.1.3`)、`mpi(2019)`，如果单独加载，请确保版本不低于它们。

- 对于 CUDA，如果您需要使用 GPU 加速，请加载 cuda工具包（建议版本不低于11.8），否则将编译纯 CPU 版。

```bash
# 加载编译器
module load cuda/11.8-share intel/2020
#此为gcc编译器，您可以加载自己的8.n版本
source /opt/rh/devtoolset-8/enable 
```

### 3. 编译lammps代码

#### step1. 为了使用 NEP模型的 GPU 版本，需要您先将 NEP 的 c++ cuda 代码编译为共享库文件。不存在CUDA则跳过该步，此时将编译纯CPU版本。
``` bash
cd lammps/src/MATPL/NEP_GPU
make clean
make
# 编译完成后您将得到一个/lammps/src/libnep_gpu.so的共享库文件
```
#### step2. 编译lammps 接口

```bash
cd lammps/src
make yes-MATPL
# 以下lammps 中常用软件，推荐在安装时顺带安装
# make yes-KSPACE
# make yes-MANYBODY
# make yes-REAXFF
# make yes-MOLECULE
# make yes-QEQ
# make yes-REPLICA
# make yes-RIGID
# make yes-MEAM
# make yes-MC
# make yes-SHOCK
make clean-all && make mpi
```

如果编译过程中找不到 `cuda_runtime.h` 头文件，请在 `src/MAKE/Makefile.mpi` 文件的 `第24行` 替换为您自己的 CUDA 路径，`/the/path/cuda/cuda-11.8`，`cuda_runtime.h` 位于该目录下的 `include` 目录下，如下所示。

```txt
CUDA_HOME = $(CUDADIR)
替换为CUDA_HOME = /the/path/cuda/cuda-11.8
```

编译完成将在窗口输出如下信息，并在lammps源码根目录生成一个env.sh文件，使用lammps前加载该文件即可。

``` txt
===========================
LAMMPS has been successfully compiled. Please load the LAMMPS environment variables before use.
You can load the environment variables by running (recommended):

    source the/path/of/lammps/env.sh

Or by executing the following commands:
    export PATH=the/path/of/lammps/src:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:the/path/of/lammps/src:/data/home/wuxingxing/anaconda3/envs/matpl-2025.3/lib/python3.11/site-packages/torch/lib:/data/home/wuxingxing/anaconda3/envs/matpl-2025.3/lib
    export LAMMPS_POTENTIALS=the/path/of/lammps/potentials
===========================
make[1]: Leaving directory `the/path/of/lammps/src/Obj_mpi'

```

### 4. lammps 加载使用
使用 lammps 需要加载它的依赖环境，加载 intel(mpi)、cuda、lammps环境变量。
``` bash
module load intel/2020 cuda/11.8-share
source /the/path/of/lammps/env.sh
```
之后即可使用如下命令启动 lammps 模拟
```
mpirun -np 4 lmp_mpi -in in.lammps
```

### 5. NEP lammps 模拟
演示案例请参考[nep_lmps_demo/]，目录结构如下所示。
```
nep_lmps_demo
    ├── nep_lmps/
    │   ├── in.lammps
    │   ├── lmp.config
    │   ├── nep_to_lmps.txt
    │   ├── runcpu.job
    │   └── rungpu.job
    ├── nep_lmps_deviation/
    │   ├── in.lammps
    │   ├── lmp.config
    │   ├── model_devi.out
    │   ├── nep0.txt
    │   ├── nep1.txt
    │   ├── nep2.txt
    │   ├── nep3.txt
    │   ├── runcpu.job
    │   └── rungpu.job
    ├── nep_test.json
    ├── nep_train.json
    └── train.job
```

### step1. 准备力场文件
NEP力场文件这里支持MatPL输出的 nep_to_lmps.txt 文件，或 GPUMD 中训练的 NEP4 力场文件。注意文件后缀必须是`.txt`

### step2. 准备输入控制文件
您需要在lammps的输入控制文件中设置如下力场

``` bash
pair_style   matpl   nep_to_lmps.txt 
pair_coeff   * *     72 8
```
- pair_style 设置力场文件路径，这里 `matpl` 未固定格式，代表使用MatPL中力场，`nep_to_lmps.txt`为力场文件路径，如 nep_lmps 目录下的案例所示。

- pair_coeff 指定待模拟结构中的原子类型对应的原子序号。例如，如果您的结构中 `1` 为 `O` 元素，`2` 为 `Hf` 元素，设置 `pair_coeff * * 8 72`即可。

- 多模型的偏差值输出，该功能一般用于主动学习采用中。您可以指定多个模型，在模拟中将使用第1个模型做MD，其他模型参与偏差值计算，如 nep_lmps_deviation 目录下的案例，此时pair_style设置为如下:
  ```txt
pair_style   matpl nep0.txt nep1.txt nep2.txt nep3.txt out_freq ${DUMP_FREQ} out_file model_devi.out
pair_coeff       * * 72 8
  ```

- 也可以把力场文件替换为 GPUMD 训练得到的 NEP4 力场文件。

### step3 启动lammps模拟
``` bash
# 加载 lammps 环境变量env.sh 文件，正确安装后，该文件位于 lammps 源码根目录下
source /the/path/of/lammps/env.sh
# 执行lammps命令
mpirun -np N lmp_mpi -in in.lammps
```
这里 N 为md中的使用的 CPU 核数，如果您的设备中存在可用的GPU资源（例如 M 张GPU卡）前编译时增加了对GPU的支持，则在运行中，N个lammps线程将平均分配到这M张卡上。我们建议您使用的 CPU 核数与您设置的 GPU 数量相同，多个线程在单个 GPU 上会由于资源竞争导致运行速度降低。

此外，lammps 接口允许跨节点以及跨节点GPU卡并行，只需要指定节点数、GPU卡数即可。
