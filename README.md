# CAWS (Cache-locality Based Adaptive Warp Scheduling)

This repository contains the source code for reproducing the experiments in the paper "Cache-locality Based Adaptive Warp Scheduling for Neural Network Acceleration on GPGPUs" in IEEE 35th International System-on-Chip Conference.

This work based on GPGPU-Sim v3.2.0

# Prerequisite

You can find GPGPU-Sim v3.2.0 in https://hub.docker.com/r/tgrogers/gpgpu-sim_regress

## Environment 

+ Ubuntu 20.04 
+ GPGPU-Sim v3.2.0
+ CUDA 11.0
+ gcc 5.5

## Benchmark

The benchmark used in this work.

+ ISPASS, https://github.com/gpgpu-sim/ispass2009-benchmarks.git
+ Parboil, http://impact.crhc.illinois.edu/parboil/parboil.aspx
+ PolyBench, http://web.cs.ucla.edu/~pouchet/software/polybench/
+ Rodinia-3.1, https://www.cs.virginia.edu/rodinia/doku.php
+ Tango, https://gitlab.com/Tango-DNNbench/Tango/-/tree/master/

# Run

Use `src` to replace the `src` directory in gpgpu-sim_distribution you download.

Use `configs/tested-cfgs/SM2_GTX480/gpgpusim.config` when you run GPGPU-Sim. We add mode `-gpgpu_scheduler caws`.

modified file:
+ shader.cc. shader.h
+ gpu-sim.cc, gpu-sim.h
+ gpu-cache.cc, gpu-cache.h
+ configs/tested-cfgs/SM2_GTX480/gpgpusim.config