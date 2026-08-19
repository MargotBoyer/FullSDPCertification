"""
Détection non-bloquante de la disponibilité CUDA.

torch.cuda.is_available() peut se bloquer indéfiniment sur certains serveurs
(contexte CUDA qui n'arrive jamais à s'initialiser, driver/torch désynchronisés —
constaté sur ce cluster : nvidia-smi répond instantanément et voit les GPU, mais
torch.cuda.is_available()/torch.zeros(1, device="cuda") restent bloqués plus de
60s). Un timeout thread-based ne suffit PAS ici : l'appel bloquant ne relâche
apparemment jamais le GIL, donc même le thread principal gèle avec lui (vérifié
empiriquement — concurrent.futures.Future.result(timeout=...) ne revient jamais).

`multiprocessing` (spawn ou fork) ne suffit PAS non plus sur ce cluster
spécifiquement : des dizaines de processus `multiprocessing.spawn_main`/
`resource_tracker` orphelins traînent depuis plusieurs jours (plusieurs
utilisateurs), signe que le mécanisme spawn lui-même reste bloqué ici — même
cause probable (ressource CUDA/IPC partagée cassée) que le hang initial.

`subprocess.run(timeout=...)` ne suffit pas non plus : après un `TimeoutExpired`,
son implémentation appelle `process.kill()` PUIS `process.communicate()` une
seconde fois SANS timeout pour drainer stdout/stderr avant de relever
l'exception — si le processus tué reste bloqué en attente noyau
ininterruptible (probable ici, même ressource cassée que le hang initial), ce
second `communicate()` bloque indéfiniment et annule tout l'intérêt du
timeout (vérifié empiriquement : hang total malgré `timeout=8`). Utiliser
`Popen` directement et ne JAMAIS rappeler `communicate()`/`wait()` après un
`kill()` évite ce piège — le parent repart immédiatement, que l'enfant tué
finisse par mourir ou reste zombie (vérifié : aucun zombie résiduel observé
avec ce pattern).
"""
from __future__ import annotations

import subprocess
import sys

_PROBE_CODE = "import torch; print('1' if torch.cuda.is_available() else '0')"


def detect_cuda_available(timeout: float = 8.0) -> bool:
    """Retourne True si un GPU CUDA utilisable est détecté dans les `timeout`
    secondes, False sinon (y compris en cas de hang, erreur, ou torch absent —
    le sous-processus est alors tué et on retombe sur CPU plutôt que de
    bloquer indéfiniment)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", _PROBE_CODE],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        out, _err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()  # ne PAS rappeler communicate()/wait() ensuite (cf. docstring module)
        # Fermer nos propres descripteurs de pipe est sûr (n'attend pas l'enfant,
        # contrairement à communicate()/wait()) et évite un ResourceWarning inoffensif
        # mais bruyant à la destruction de l'objet Popen.
        if proc.stdout:
            proc.stdout.close()
        if proc.stderr:
            proc.stderr.close()
        return False
    return proc.returncode == 0 and out.strip() == "1"


_patched = False


def patch_torch_cuda_is_available(timeout: float = 8.0) -> bool:
    """Sonde une seule fois (résultat mis en cache) puis remplace
    torch.cuda.is_available par une fonction qui renvoie ce résultat en cache
    (donc instantanée, sans nouveau risque de hang) — pour TOUS les appelants
    du process, pas seulement l'appel initial dans certification_problem.py.
    Idempotent (un seul probe réel même si appelée plusieurs fois)."""
    global _patched
    import torch

    if _patched:
        return torch.cuda.is_available()

    available = detect_cuda_available(timeout=timeout)
    torch.cuda.is_available = lambda: available
    _patched = True
    return available
