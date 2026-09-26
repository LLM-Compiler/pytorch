source ~/miniconda3/etc/profile.d/conda.sh
conda activate triton_cpu_integrated
export LD_LIBRARY_PATH=/opt/intel/oneapi/dnnl/2023.1.0/cpu_gomp/lib:$LD_LIBRARY_PATH
cd /home/shavin/llm-cpu-compiler/pytorch
R=bench_2026-09-25
for NL in 1 2; do
  PASSES=std NLAYERS=$NL WS=4 python $R/numa_check.py
  for WS in 2 4; do
    for P in mlp attn both; do
      PASSES=$P NLAYERS=$NL WS=$WS python $R/numa_check.py || echo "FAILED $P w$WS L$NL"
    done
  done
done
echo MATRIXDONE
