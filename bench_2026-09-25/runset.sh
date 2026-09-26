set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25
# A: zero fc2 weight -> MLP contributes only its (zero) bias; rest of graph unchanged
ZERO=fc2 python $R/mlp_gpt2.py std  4
ZERO=fc2 python $R/mlp_gpt2.py numa 4
# B: different d_ffn so shard_d != h_in (breaks the 3072/4 == 768 coincidence)
DFFN=4096 python $R/mlp_gpt2.py std  4
DFFN=4096 python $R/mlp_gpt2.py numa 4
# C: world size 2
python $R/mlp_gpt2.py std  2
python $R/mlp_gpt2.py numa 2
echo ALLDONE
