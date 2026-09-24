#!/bin/bash
# Script pour lancer un run sur jean-zay
# Configuration - À modifier selon tes paramètres
JEAN_ZAY_USER="uvq13au"
JEAN_ZAY_HOST="jean-zay.idris.fr"
LOCAL_PROJECT_DIR="/share/homes/boyerma/FastSDPCertification"
REMOTE_WORK_DIR="/lustre/fswork/projects/rech/llc/uvq13au"
REMOTE_PROJECT_DIR="$REMOTE_WORK_DIR/FastSDPCertification"  # Nom de ton projet/dossier sur Jean-Zay

# Dossiers à synchroniser
FOLDERS_TO_SYNC=("src" "data" "notebooks" "config")

# Sources ConicBundle (Helmberg/Kiwiel), nécessaires pour recompiler libcb.a puis
# src/miqcr_bridge/conicbundle_native/libcb_native.so sur Jean-Zay (le .so compilé
# localement ne doit pas être utilisé tel quel : autre distro/toolchain). On exclut
# la doc Doxygen (html/) et les objets déjà compilés (DEBU.*/OPTI.*/lib/, ABI
# potentiellement incompatible) : on force un rebuild propre côté Jean-Zay via
# `make -C Miqcr-1.0_triang_sbb_gurobiter_QCP5/ConicBundle` puis
# `make -C src/miqcr_bridge/conicbundle_native`.
CONICBUNDLE_PARENT="Miqcr-1.0_triang_sbb_gurobiter_QCP5"
CONICBUNDLE_DIR="$CONICBUNDLE_PARENT/ConicBundle"

# Couleurs pour les messages
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages colorés
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Vérifications préliminaires
if [ ! -d "$LOCAL_PROJECT_DIR" ]; then
    log_error "Le dossier local $LOCAL_PROJECT_DIR n'existe pas !"
    exit 1
fi

# Vérification que les dossiers à synchroniser existent
for folder in "${FOLDERS_TO_SYNC[@]}"; do
    if [ ! -d "$LOCAL_PROJECT_DIR/$folder" ]; then
        log_warning "Le dossier $LOCAL_PROJECT_DIR/$folder n'existe pas, il sera ignoré"
    fi
done

# SSH ControlMaster : une seule connexion TCP réutilisée pour tous les appels
SSH_CONTROL="/tmp/ssh_cm_jz_$$"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SSH_CONTROL -o ControlPersist=120 -o BatchMode=yes"
ssh_jz() { ssh $SSH_OPTS "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" "$@"; }
cleanup_ssh() { ssh -O exit -o ControlPath=$SSH_CONTROL "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" 2>/dev/null; }
trap cleanup_ssh EXIT

# Test de connexion SSH (ouvre la connexion maître)
log_info "Test de connexion à Jean-Zay..."
if ! ssh_jz -o ConnectTimeout=5 exit 2>/dev/null; then
    log_error "Impossible de se connecter à Jean-Zay. Vérifie tes clés SSH et ta connexion."
    exit 1
fi
log_success "Connexion SSH OK"

# ─────────────────────────────────────────────────────────────────────────────
# MODE --resume
# Usage : bash run_fastsdp_jean_zay.sh --resume <chemin_dossier_local_ou_relatif>
# Exemple :
#   bash scripts/run_fastsdp_jean_zay.sh --resume results/benchmark/9x100-0.026/2026_06_01_...
# ─────────────────────────────────────────────────────────────────────────────
if [ "$1" = "--resume" ]; then
    LOCAL_RUN_FOLDER="$2"
    if [ -z "$LOCAL_RUN_FOLDER" ]; then
        log_error "--resume nécessite un chemin de dossier."
        echo "Usage: bash run_fastsdp_jean_zay.sh --resume <chemin>"
        exit 1
    fi

    # Normaliser en chemin absolu local
    if [[ "$LOCAL_RUN_FOLDER" != /* ]]; then
        LOCAL_RUN_FOLDER="$LOCAL_PROJECT_DIR/$LOCAL_RUN_FOLDER"
    fi
    LOCAL_RUN_FOLDER=$(realpath "$LOCAL_RUN_FOLDER" 2>/dev/null || echo "$LOCAL_RUN_FOLDER")

    # Mapper vers le chemin distant
    if [[ "$LOCAL_RUN_FOLDER" == "$LOCAL_PROJECT_DIR"* ]]; then
        REL_PATH="${LOCAL_RUN_FOLDER#$LOCAL_PROJECT_DIR/}"
        REMOTE_RUN_FOLDER="$REMOTE_PROJECT_DIR/$REL_PATH"
    else
        log_error "Le dossier doit être sous $LOCAL_PROJECT_DIR"
        exit 1
    fi

    log_info "Reprise du run : $REMOTE_RUN_FOLDER"

    # Synchronisation du code
    log_info "Synchronisation du code vers Jean-Zay..."
    for folder in "${FOLDERS_TO_SYNC[@]}"; do
        if [ -d "$LOCAL_PROJECT_DIR/$folder" ]; then
            rsync -az -e "ssh $SSH_OPTS" "$LOCAL_PROJECT_DIR/$folder" "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_PROJECT_DIR"
        fi
    done
    rsync -az -e "ssh $SSH_OPTS" "$LOCAL_PROJECT_DIR/scripts" "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_PROJECT_DIR"
    if [ -d "$LOCAL_PROJECT_DIR/$CONICBUNDLE_DIR" ]; then
        ssh_jz "mkdir -p '$REMOTE_PROJECT_DIR/$CONICBUNDLE_PARENT'"
        rsync -az -e "ssh $SSH_OPTS" \
            --exclude='html/' --exclude='DEBU.*/' --exclude='OPTI.*/' --exclude='lib/' \
            "$LOCAL_PROJECT_DIR/$CONICBUNDLE_DIR" "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_PROJECT_DIR/$CONICBUNDLE_PARENT/"
    fi
    log_success "Synchronisation terminée."

    # Chercher les sous-dossiers part_* sur Jean-Zay
    PART_FOLDERS=$(ssh_jz "ls -d $REMOTE_RUN_FOLDER/part_* 2>/dev/null" || true)

    if [ -n "$PART_FOLDERS" ]; then
        # Run divisé : un job de reprise par chunk
        log_info "Run divisé détecté — soumission d'un job par chunk :"
        while IFS= read -r part_folder; do
            [ -z "$part_folder" ] && continue
            log_info "  → $part_folder"
            ssh_jz "cd $REMOTE_PROJECT_DIR && sbatch --export=RESUME_FOLDER=$part_folder scripts/run_fastsdp_job.slurm"
        done <<< "$PART_FOLDERS"
    else
        # Run simple : un seul job
        log_info "Soumission du job de reprise pour $REMOTE_RUN_FOLDER"
        ssh_jz "cd $REMOTE_PROJECT_DIR && sbatch --export=RESUME_FOLDER=$REMOTE_RUN_FOLDER scripts/run_fastsdp_job.slurm"
    fi

    log_success "Job(s) de reprise soumis à Jean-Zay."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# MODE NORMAL
# ─────────────────────────────────────────────────────────────────────────────
# Extraction des flags --start/--end (peuvent apparaître n'importe où après les
# arguments positionnels), le reste alimente NETWORK_NAME / NAME_RUN / EPSILON.
CLI_START=""
CLI_END=""
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --start)
            CLI_START="$2"
            shift 2
            ;;
        --end)
            CLI_END="$2"
            shift 2
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]}"

NETWORK_NAME=$1
NAME_RUN=$2
EPSILON=$3

if [ -z "$NETWORK_NAME" ] || [ -z "$NAME_RUN" ]; then
  echo "Erreur: paramètres manquants."
  echo "Usage: bash run_fastsdp_jean_zay.sh <NETWORK_NAME> <NAME_RUN> [EPSILON] [--start N] [--end M]"
  echo "       bash run_fastsdp_jean_zay.sh --resume <chemin_dossier>"
  exit 1
fi

if { [ -n "$CLI_START" ] && [ -z "$CLI_END" ]; } || { [ -z "$CLI_START" ] && [ -n "$CLI_END" ]; }; then
  log_error "--start et --end doivent être fournis ensemble."
  exit 1
fi

# Détection automatique de EPSILON si non fourni
if [ -z "$EPSILON" ]; then
  case $NETWORK_NAME in
    6x100|9x100) EPSILON=0.026 ;;
    6x200|9x200) EPSILON=0.015 ;;
    MLP-ADV)     EPSILON=0.1   ;;
    *) echo "Erreur: EPSILON non fourni pour réseau inconnu."; exit 1 ;;
  esac
fi

# Préfixe MNIST si besoin
if [[ "$NETWORK_NAME" =~ ^(6x100|6x200|9x100|9x200|MLP-ADV)$ ]]; then
  NETWORK_NAME="mnist-$NETWORK_NAME"
fi

echo "Projet bien reçu avec les paramètres :"

# Envoie le script SLURM et le projet à Jean Zay (si ce n'est pas déjà fait)
log_info "Synchronisation du projet vers Jean-Zay..."
for folder in "${FOLDERS_TO_SYNC[@]}"; do
    if [ -d "$LOCAL_PROJECT_DIR/$folder" ]; then
        rsync -az -e "ssh $SSH_OPTS" "$LOCAL_PROJECT_DIR/$folder" "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_PROJECT_DIR"
    fi
done

rsync -az -e "ssh $SSH_OPTS" "$LOCAL_PROJECT_DIR/scripts" "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_PROJECT_DIR"

if [ -d "$LOCAL_PROJECT_DIR/$CONICBUNDLE_DIR" ]; then
    ssh_jz "mkdir -p '$REMOTE_PROJECT_DIR/$CONICBUNDLE_PARENT'"
    rsync -az -e "ssh $SSH_OPTS" \
        --exclude='html/' --exclude='DEBU.*/' --exclude='OPTI.*/' --exclude='lib/' \
        "$LOCAL_PROJECT_DIR/$CONICBUNDLE_DIR" "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_PROJECT_DIR/$CONICBUNDLE_PARENT/"
fi

log_success "Synchronisation terminée."

YAML_PATH="$LOCAL_PROJECT_DIR/config/${NETWORK_NAME}.yaml"
DIVIDE_RUN=$(python -c "import yaml; c=yaml.safe_load(open('$YAML_PATH')); print(c.get('divide_run', 1))" 2>/dev/null || echo 1)

if [ -n "$CLI_START" ] && [ -n "$CLI_END" ]; then
    # --start/--end explicites en ligne de commande : un seul job, on bypasse
    # le découpage automatique via divide_run.
    log_info "Plage explicite demandée : samples [$CLI_START, $CLI_END)"
    REMOTE_COMMAND="cd $REMOTE_PROJECT_DIR && sbatch --export=NETWORK_NAME=$NETWORK_NAME,NAME_RUN=$NAME_RUN,EPSILON=$EPSILON,START=$CLI_START,END=$CLI_END scripts/run_fastsdp_job.slurm"
    log_info "Soumission du job SLURM à Jean-Zay..."
    ssh_jz "$REMOTE_COMMAND"
    log_success "Job soumis à Jean-Zay."
elif [ "$DIVIDE_RUN" -gt 1 ]; then
    NUM_SAMPLES=$(python -c "import yaml; c=yaml.safe_load(open('$YAML_PATH')); print(c['data']['num_samples'])")
    DATE_PREFIX=$(date +%Y_%m_%d_%Hh%M_%Ss)
    NAME_RUN_FULL="${DATE_PREFIX}_${NAME_RUN}"
    CHUNK_SIZE=$(( (NUM_SAMPLES + DIVIDE_RUN - 1) / DIVIDE_RUN ))

    log_info "Découpage : $DIVIDE_RUN chunks de ~$CHUNK_SIZE samples sur $NUM_SAMPLES total"
    log_info "Run partagé : $NAME_RUN_FULL"

    for ((chunk=0; chunk<DIVIDE_RUN; chunk++)); do
        START=$((chunk * CHUNK_SIZE))
        END=$(( (chunk + 1) * CHUNK_SIZE ))
        if [ "$END" -gt "$NUM_SAMPLES" ]; then END=$NUM_SAMPLES; fi
        if [ "$START" -ge "$NUM_SAMPLES" ]; then break; fi

        log_info "Soumission chunk $chunk: samples [$START, $END)"
        REMOTE_COMMAND="cd $REMOTE_PROJECT_DIR && sbatch --export=NETWORK_NAME=$NETWORK_NAME,NAME_RUN=$NAME_RUN_FULL,EPSILON=$EPSILON,START=$START,END=$END scripts/run_fastsdp_job.slurm"
        ssh_jz "$REMOTE_COMMAND"
    done
    log_success "$DIVIDE_RUN jobs soumis à Jean-Zay."
else
    REMOTE_COMMAND="cd $REMOTE_PROJECT_DIR && sbatch --export=NETWORK_NAME=$NETWORK_NAME,NAME_RUN=$NAME_RUN,EPSILON=$EPSILON scripts/run_fastsdp_job.slurm"
    log_info "Soumission du job SLURM à Jean-Zay..."
    ssh_jz "$REMOTE_COMMAND"
    log_success "Job soumis à Jean-Zay."
fi
