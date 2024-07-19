#!/bin/bash

timestamp=$(date +"%Y%m%d%H%M%S%N")  # %N for nanoseconds

random_suffix=$(($RANDOM % 1000))  # Generates a random number between 0 and 999

jobname="automatic_batchsize_${timestamp}_${random_suffix}"

qsub -N "${jobname}" -o "/projectnb/tianlabdl/rsyed/cache_dataloading/logs/${jobname}.qlog" "batch.qsub"
