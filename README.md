

## Lammps-fortran 编译安装
此版本为 MatPL 的 NN、Linear 模型 lammps 力场接口。

lammps 源码安装需要您下载 lammps 源码、加载编译器、编译源码，过程如下所示。

### 1. 准备 MatPL 力场接口和 lammps 源码

#### MATPL 力场接口源码
MATPL 力场接口源码位于 [MatPL 源码目录/src/lmps] 下，您也可以通过 github 下载 MATPL 力场接口源码，或者下载 release 包。
- 通过 github 或 gitee clone 源码:
```bash
git clone -b fortran git@github.com:LonxunQuantum/lammps-MatPL.git
或
git clone -b fortran https://gitee.com/pfsuo/lammps-MatPL.git
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
├── lmps-examples/
└── README.md
```

#### Lammps 源码

lammps 源码请访问 [lammps github 仓库](https://github.com/lammps/lammps/tree/stable#) 下载，这里推荐下载 `stable 版本`。

#### lammps 源码目录下设置力场文件

- 复制 Makefile.mpi 文件到 lammps/src/MAKE/目录下

- 复制 MATPL 目录到 lammps/src/目录下

### 2. 加载编译环境
首先检查 `intel/2020`，`gcc8.n`是否加载；

- 对于 `intel/2020`编译套件，使用了它的 `ifort` 和 `icc` 编译器(`19.1.3`)、`mpi(2019)`、`mkl库(2020)`，如果单独加载，请确保版本不低于它们。

```bash
# 加载编译器
module load intel/2020
#此为gcc编译器，您可以加载自己的8.n版本
source /opt/rh/devtoolset-8/enable 
```

### 3. 编译lammps代码

#### step1. 编译fortran 力场库文件
``` bash
cd lammps/src/MATPL/fortran_code
make clean
make
# 编译完成后您将得到一个/lammps/src/MATPL/f2c_calc_energy_force.a 文件
```
#### step2. 编译lammps 接口

```bash
cd lammps/src
make yes-MATPL
# 以下lammps 中常用软件，推荐在安装时顺带安装
make yes-KSPACE
make yes-MANYBODY
make yes-REAXFF
make yes-MOLECULE
make yes-QEQ
make yes-REPLICA
make yes-RIGID
make yes-MEAM
make yes-MC
make yes-SHOCK
# 开始编译
make clean-all
make mpi -j4 mode=shared # 这里4为并行编译数量，shared为编译出一个共享库文件，可以用于python相关操作中
```

编译完成将在窗口输出如下信息，并在lammps源码根目录生成一个env.sh文件，使用lammps前加载该文件即可。

``` txt
===========================
LAMMPS has been successfully compiled. Please load the LAMMPS environment variables before use.
You can load the environment variables by running (recommended):

    source the/path/of/lammps/env.sh

Or by executing the following commands:
    export PATH=the/path/of/lammps/src:$PATH
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:the/path/of/lammps/src
    export LAMMPS_POTENTIALS=the/path/of/lammps/potentials
===========================
make[1]: Leaving directory `the/path/of/lammps/src/Obj_shared_mpi'

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

### 5. NN lammps 模拟
演示案例请参考[nn_lmps/]，目录结构如下所示。
```
lmps-examples
   ├── linear_lmps
   │   ├── forcefield.ff
   │   ├── in.lammps
   │   ├── lmp.config
   │   └── runcpu.job
   └── nn_lmps
      ├── forcefield.ff
      ├── in.lammps
      ├── lmp.config
      └── runcpu.job
```

### step1. 准备力场文件
NN 力场文件请参考 [MatPL 在线文档](http://doc.lonxun.com/PWMLFF/)。

### step2. 准备输入控制文件
您需要在lammps的输入控制文件中设置如下力场，NN 力场的输入不同于 DP或者NEP，如下所示，这里以 `lmps-examples/nn_lmps` 为例，该例为一个铜的BULK结构在300K下的模拟：

``` bash
pair_style   matpl 
pair_coeff   * * 3 1 forcefield.ff 29
```
- pair_style 设置使用 matpl 力场

- pair_coeff 设置力场文件和原子类型。这里 `3` 表示使用 Neural Network 模型产生的力场，如果使用 Liear 力场，请设置为`1`；第二个数字 `1` 表示读取 1 个力场文件，forcefield.ff为 MatPL 生成的力场文件名称，29 为 Cu 的原子序数

### step3 启动lammps模拟
``` bash
# 加载 lammps 环境变量env.sh 文件，正确安装后，该文件位于 lammps 源码根目录下
source /the/path/of/lammps/env.sh
# 执行lammps命令
mpirun -np N lmp_mpi -in in.lammps
```
