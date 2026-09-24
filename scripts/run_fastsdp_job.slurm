#!/bin/bash
#SBATCH --job-name=fast_sdp
#SBATCH -A llc@cpu                      # Compte CPU (MOSEK est CPU-only)
#SBATCH --partition=cpu_p1              # Partition CPU
#SBATCH --cpus-per-task=40             # Nœud entier : 40 CPUs = ~192 Go RAM sur cpu_p1 (seul moyen d'avoir plus de mémoire sur Jean-Zay)
#SBATCH --time=20:00:00                 # Temps max par job (limite cpu_p1 sur Jean-Zay)
#SBATCH --chdir=/lustre/fswork/projects/rech/llc/uvq13au/FastSDPCertification
#SBATCH --output=slurm-%x-%j.out       # Log SLURM standard (à la racine du projet)
#SBATCH --error=slurm-%x-%j.err        # Log SLURM erreur

# Charge les modules nécessaires
module purge
source ~/.bashrc
module load pytorch-gpu/py3/2.0.0
module unload openmpi
module load openmpi/4.1.1


# Active ton venv
source /lustre/fswork/projects/rech/llc/uvq13au/venvs/fastsdp/bin/activate

# Recompile systematiquement le pont ConicBundle natif (src/miqcr_bridge/conicbundle_native/)
# avant de lancer le run : le .so ne doit jamais etre reutilise d'une machine a l'autre
# (ABI/toolchain), et recompiler coute ~2-3 min contre un job de plusieurs heures.
# Sans effet si la conic bundle native n'est pas utilisee dans le yaml.
if [ -d "Miqcr-1.0_triang_sbb_gurobiter_QCP5/ConicBundle" ]; then
    echo "Recompilation de ConicBundle (MODE=OPTI) + du pont natif..."
    module load gcc
    make -C Miqcr-1.0_triang_sbb_gurobiter_QCP5/ConicBundle MODE=OPTI \
        && make -C src/miqcr_bridge/conicbundle_native \
        || { echo "ERREUR: echec de compilation de ConicBundle/libcb_native.so"; exit 1; }
fi

if [ -n "$RESUME_FOLDER" ]; then
    # ── MODE REPRISE ──────────────────────────────────────────────────────────
    # Les logs vont dans le dossier existant (nouveau fichier par job ID)
    exec > >(tee -a "$RESUME_FOLDER/slurm-resume-${SLURM_JOB_ID}.out")
    exec 2> >(tee -a "$RESUME_FOLDER/slurm-resume-${SLURM_JOB_ID}.err" >&2)

    echo "=== REPRISE ==="
    echo "Dossier : $RESUME_FOLDER"
    echo "Job ID  : $SLURM_JOB_ID"

    python src/certification_problem.py --resume "$RESUME_FOLDER"

else
    # ── MODE NORMAL ───────────────────────────────────────────────────────────
    DATE=$(date +%Y_%m_%d_%Hh%M_%Ss)
    LOG_DIR="results/benchmark/${NETWORK_NAME}-${EPSILON}/${DATE}_${NAME_RUN}"
    mkdir -p "$LOG_DIR"

    exec > >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.out")
    exec 2> >(tee -a "$LOG_DIR/slurm-${SLURM_JOB_ID}.err" >&2)

    echo "Network: $NETWORK_NAME"
    echo "Run name: $NAME_RUN"
    echo "Job ID: $SLURM_JOB_ID"

    if [ -n "$START" ] && [ -n "$END" ]; then
        python src/certification_problem.py "$NETWORK_NAME" "$NAME_RUN" --start "$START" --end "$END"
    else
        python src/certification_problem.py "$NETWORK_NAME" "$NAME_RUN"
    fi
fi
