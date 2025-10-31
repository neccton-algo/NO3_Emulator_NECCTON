#!/bin/bash
if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please provide the year."
    exit 1
fi

YEAR=$1
CURR_DIR=$(pwd)
SCRIPT_DIR=${CURR_DIR}/files_tobe/
WORK_DIR=${CURR_DIR}/${YEAR}

mkdir -p $WORK_DIR
cp ${SCRIPT_DIR}/pred_NO3_optm_1core.py ${WORK_DIR}/.
sed "s/_YYYY_/${YEAR}/g" ${SCRIPT_DIR}/run.slurm > ${WORK_DIR}/run.slurm && chmod +x ${WORK_DIR}/run.slurm
echo "Directory for year $YEAR created, and scripts copied and modified."
