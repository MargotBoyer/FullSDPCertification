#!/bin/bash
# Script pour copier le code et les données vers le serveur Jean-Zay via rsync
# Configuration - À modifier selon tes paramètres
JEAN_ZAY_USER="uvq13au"
JEAN_ZAY_HOST="jean-zay.idris.fr"
LOCAL_PROJECT_DIR="/share/homes/boyerma/FastSDPCertification"
REMOTE_PROJECT_DIR="FastSDPCertification"  # Nom de ton projet/dossier sur Jean-Zay

# Dossiers à synchroniser
FOLDERS_TO_SYNC=("src" "data" "notebooks" "config")

# Sources ConicBundle (Helmberg/Kiwiel), nécessaires pour recompiler libcb.a puis
# src/miqcr_bridge/conicbundle_native/libcb_native.so sur Jean-Zay (le .so compilé
# localement ne doit pas être utilisé tel quel : autre distro/toolchain). On exclut
# la doc Doxygen (html/, ~5,6 Mo) et les objets déjà compilés (DEBU.*/OPTI.*/lib/,
# ABI potentiellement incompatible) : on force un rebuild propre côté Jean-Zay via
# `make -C Miqcr-1.0_triang_sbb_gurobiter_QCP5/ConicBundle` puis
# `make -C src/miqcr_bridge/conicbundle_native`.
CONICBUNDLE_DIR="Miqcr-1.0_triang_sbb_gurobiter_QCP5/ConicBundle"

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

# Test de connexion SSH
log_info "Test de connexion à Jean-Zay..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" exit 2>/dev/null; then
    log_error "Impossible de se connecter à Jean-Zay. Vérifie tes clés SSH et ta connexion."
    exit 1
fi
log_success "Connexion SSH OK"

# Récupération du chemin $WORK depuis Jean-Zay
log_info "Récupération du chemin \$WORK depuis Jean-Zay..."
REMOTE_WORK_PATH=/lustre/fswork/projects/rech/llc/uvq13au     #$(ssh "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" 'echo $WORK')

# if [ -z "$REMOTE_WORK_PATH" ]; then
#     log_error "Impossible de récupérer le chemin \$WORK depuis Jean-Zay"
#     exit 1
# fi

log_info "Chemin \$WORK détecté: $REMOTE_WORK_PATH"

# Création du dossier distant si nécessaire
log_info "Création du dossier distant si nécessaire..."
ssh "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" "mkdir -p '$REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR'"

# Options rsync
RSYNC_OPTIONS=(
    -avz                    # archive, verbose, compress
    --delete               # supprime les fichiers qui n'existent plus en local
    --exclude='.git/'      # exclut le dossier .git
    --exclude='*.pyc'      # exclut les fichiers Python compilés
    --exclude='__pycache__/' # exclut les caches Python
    --exclude='.DS_Store'  # exclut les fichiers macOS
    --exclude='*.tmp'      # exclut les fichiers temporaires
    --exclude='*.log'      # exclut les logs (optionnel)
    --progress             # affiche le progrès
    --human-readable       # tailles lisibles
)

# Synchronisation
log_info "Synchronisation des dossiers vers Jean-Zay..."
log_info "Destination: $JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR/"

# Synchroniser chaque dossier individuellement
sync_success=true
for folder in "${FOLDERS_TO_SYNC[@]}"; do
    if [ -d "$LOCAL_PROJECT_DIR/$folder" ]; then
        log_info "Synchronisation du dossier '$folder'..."
        
        if rsync "${RSYNC_OPTIONS[@]}" \
            "$LOCAL_PROJECT_DIR/$folder/" \
            "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR/$folder/"; then
            log_success "Dossier '$folder' synchronisé avec succès"
        else
            log_error "Erreur lors de la synchronisation du dossier '$folder'"
            sync_success=false
        fi
    else
        log_warning "Dossier '$folder' ignoré (n'existe pas localement)"
    fi
done

# Sources ConicBundle (pas dans FOLDERS_TO_SYNC : hors src/, cf. commentaire plus haut)
if [ -d "$LOCAL_PROJECT_DIR/$CONICBUNDLE_DIR" ]; then
    log_info "Synchronisation des sources ConicBundle ('$CONICBUNDLE_DIR')..."
    ssh "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" "mkdir -p '$REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR/$CONICBUNDLE_DIR'"
    if rsync "${RSYNC_OPTIONS[@]}" \
        --exclude='html/' --exclude='DEBU.*/' --exclude='OPTI.*/' --exclude='lib/' \
        "$LOCAL_PROJECT_DIR/$CONICBUNDLE_DIR/" \
        "$JEAN_ZAY_USER@$JEAN_ZAY_HOST:$REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR/$CONICBUNDLE_DIR/"; then
        log_success "Sources ConicBundle synchronisées avec succès"
    else
        log_error "Erreur lors de la synchronisation des sources ConicBundle"
        sync_success=false
    fi
else
    log_warning "Dossier '$CONICBUNDLE_DIR' ignoré (n'existe pas localement)"
fi

if [ "$sync_success" = true ]; then
    log_success "Synchronisation terminée avec succès !"
else
    log_error "Certaines synchronisations ont échoué !"
    exit 1
fi
    
    # Affichage des informations sur l'espace distant
    log_info "Vérification de l'espace disque sur Jean-Zay..."
    ssh "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" "
        echo 'Espace dans \$WORK:'
        df -h \$WORK 2>/dev/null || echo 'Impossible de vérifier \$WORK'
        echo
        echo 'Contenu du dossier synchronisé:'
        ls -la \$WORK/$REMOTE_PROJECT_DIR/ 2>/dev/null || echo 'Dossier non trouvé'
    "


# Mode interactif pour des actions supplémentaires
echo
read -p "Veux-tu ouvrir une session SSH sur Jean-Zay maintenant ? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Connexion à Jean-Zay..."
    ssh -t "$JEAN_ZAY_USER@$JEAN_ZAY_HOST" "cd $REMOTE_WORK_PATH/$REMOTE_PROJECT_DIR && bash"
fi