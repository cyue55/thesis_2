#!/bin/bash

OPTS=$(getopt \
    --options fvq:c:m:W:s:M:g:p:bx:B:w:S:V: \
    --longoptions force,verbose,queue:,cores:,memory:,walltime:,select:,model:,bottleneck,extraargs:,schedule:,dependency:,script:,venv: \
    -- "$@"
)
if [ $? -ne 0 ]; then exit 1; fi
eval set -- "$OPTS"

FORCE=false
VERBOSE=false
QUEUE="gpuv100"
CORES=4
MEMORY=4
WALLTIME=24
SELECT=""
MODEL=""
BOTTLENECK=false
EXTRAARGS=""
SCHEDULE=""
DEPENDENCY=""
SCRIPT="train.py"
VENV="venv"

while true; do
  case "$1" in
    -f | --force ) FORCE=true; shift ;;
    -v | --verbose ) VERBOSE=true; shift ;;
    -q | --queue ) QUEUE="$2"; shift; shift ;;
    -c | --cores ) CORES="$2"; shift; shift ;;
    -m | --memory ) MEMORY="$2"; shift; shift ;;
    -W | --walltime ) WALLTIME="$2"; shift; shift ;;
    -s | --select ) SELECT="$2"; shift; shift ;;
    -M | --model ) MODEL="$2"; shift; shift ;;
    -b | --bottleneck ) BOTTLENECK=true; shift ;;
    -x | --extraargs ) EXTRAARGS="$2"; shift; shift ;;
    -B | --schedule ) SCHEDULE="$2"; shift; shift ;;
    -w | --dependency ) DEPENDENCY="$2"; shift; shift ;;
    -S | --script ) SCRIPT="$2"; shift; shift ;;
    -V | --venv ) VENV="$2"; shift; shift ;;
    -- ) shift; break ;;
  esac
done

if [ $# -eq 0 ]
then
    echo "No input provided"
    exit 1
fi

if [ "${QUEUE}" != "hpc" ] && [ "${QUEUE}" != "gpuv100" ] && [ "${QUEUE}" != "gpua100" ]
then
    echo "ERROR: requested wrong queue: ${QUEUE}"
    exit 1
fi

JOBFILE="lsf/template.sh"
echo "#!/bin/bash" > ${JOBFILE}
echo "#BSUB -q ${QUEUE}" >> ${JOBFILE}
echo "#BSUB -J jobname" >> ${JOBFILE}
echo "#BSUB -n ${CORES}" >> ${JOBFILE}
echo "#BSUB -W ${WALLTIME}:00" >> ${JOBFILE}
echo "#BSUB -R \"rusage[mem=${MEMORY}GB]\"" >> ${JOBFILE}
echo "#BSUB -R \"span[hosts=1]\"" >> ${JOBFILE}
echo "#BSUB -oo lsf/logs/%J.out" >> ${JOBFILE}
echo "#BSUB -eo lsf/logs/%J.err" >> ${JOBFILE}

if [ "${QUEUE}" = "gpuv100" ] || [ "${QUEUE}" = "gpua100" ]
then
    echo "#BSUB -gpu \"num=1:mode=exclusive_process\"" >> ${JOBFILE}
fi

if [ "${SELECT}" != "" ]
then
    echo "#BSUB -R \"select[${SELECT}]\"" >> ${JOBFILE}
fi

if [ "${MODEL}" != "" ]
then
    echo "#BSUB -R \"select[model == ${MODEL}]\"" >> ${JOBFILE}
fi

if [ "${SCHEDULE}" != "" ]
then
    echo "#BSUB -b ${SCHEDULE}" >> ${JOBFILE}
fi

if [ "${DEPENDENCY}" != "" ]
then
    echo "#BSUB -w \"${DEPENDENCY}\"" >> ${JOBFILE}
fi

echo "COMMAND" >> ${JOBFILE}

if [ ${VERBOSE} = true ]
then
    echo "The following job template was created:"
    echo "---Beginning of file---"
    cat ${JOBFILE}
    echo "---End of file---"
fi

for INPUT in "$@"
do
    if [ -f "${INPUT}/losses.npz" ] && [ ${FORCE} = false ]
    then
        echo "model already trained: ${INPUT}"
        continue
    fi
    PRECOMMAND=""
    COMMAND="${SCRIPT} ${INPUT}"
    if [ ${FORCE} = true ]
    then
        COMMAND="${COMMAND} -f"
    fi
    if [ "${EXTRAARGS}" != "" ]
    then
        COMMAND="${COMMAND} ${EXTRAARGS}"
    fi
    if [ ${BOTTLENECK} == true ]
    then
        COMMAND="-m torch.utils.bottleneck ${COMMAND}"
    fi
    COMMAND="python ${COMMAND}"
    COMMAND="module load gcc/14.2.0-binutils-2.43\nmodule load cuda/12.6.2\nsource ${VENV}/bin/activate\n${COMMAND}"
    if [ ${VERBOSE} = true ]
    then
        echo "Submitting the job template with the following as COMMAND":
        echo "---Beginning of COMMAND---"
        printf "${COMMAND}\n"
        echo "---End of COMMAND---"
    fi
    sed "s|COMMAND|${COMMAND}|g" "${JOBFILE}" | bsub
done
