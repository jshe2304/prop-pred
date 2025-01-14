#!/bin/sh

if [[ "$1" == "cpu" ]]; then

	sinteractive \
		--account=pi-andrewferguson \
		--partition=caslake \
		--nodes=1 \
		--ntasks-per-node=8 \
		--mem-per-cpu=2000

elif [[ "$1" == "gpu" ]]; then

	sinteractive \
		--account=pi-andrewferguson \
		--partition=gpu \
		--gres=gpu:1 \
		--nodes=1 \
		--ntasks-per-node=8

else

	echo "Unknown argument."

fi
