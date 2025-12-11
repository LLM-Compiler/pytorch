#!/bin/bash

#
# Build Triton from local source
# Tharindu Patabandi <tharindu@protonmail.com>
#

# check if found locally at the same parent level as pytorch
# We expect the three directories (i.e, pytorch, triton-cpu, llvm-project) to be at the same dir level
if [[ -d ../triton-cpu ]]; then
	# ideally, we should check if this is the Triton verison we need
	# i.e, check triton's git remote and HEAD
	echo "Triton source found"
	cd ../triton-cpu
	
else
	# else, pull from github
	echo "Cloning triton from https://github.com/LLM-Compiler/triton-cpu"
	git clone https://github.com/LLM-Compiler/triton-cpu ../triton-cpu
	cd ../triton-cpu
fi

#TODO: call triton build here 
chmod +x ./build_triton.sh

bash ./build_triton.sh

