/**
 * miqcr_pyapi.c
 *
 * Wrapper C exposé comme shared library pour l'interface Python/ctypes.
 * Peuple les structures globales MIQCR (qp, psdp) depuis des tableaux numpy
 * et exécute la phase SDP (Conic Bundle → compute_alpha_beta_sbb_miqcr).
 *
 * API exportée :
 *   void miqcr_populate_qp(n, nb_int, m, p, mq, pq,
 *                           Q, c, u, l, A, b, D, e, Aq, bq, Dq, eq, cons)
 *   void miqcr_compute_betas(out_beta, out_alphaq, out_alphabisq, out_sol_sdp)
 *   void miqcr_set_param(const char *name, double value)
 *
 * Compilation (voir Makefile.pyapi) :
 *   gcc -shared -fPIC -o libmiqcr_pyapi.so miqcr_pyapi.c <autres .o> <libs>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "quad_prog.h"      /* MIQCP, SDP, ij2k, ijk2l, new_miqcp, … */
#include "utilities.h"      /* alloc_vector_d, alloc_matrix_d, alloc_matrix_3d, … */
#include "parameters.h"     /* FACTOR, EPS_TERM_CB, NB_MAX_ITER, … */
#include "solver_sdp_mixed.h" /* compute_alpha_beta_sbb_miqcr, create_sdp_mixed */
#include "in_out.h"         /* déclarations pour init_param */

/* ---------------------------------------------------------------------- */
/* Variables globales requises par la lib MIQCR (définies ici pour la .so) */
/* ---------------------------------------------------------------------- */

MIQCP  qp    = NULL;
SDP    psdp;           /* struct (pas pointeur) */
double sol_sdp = 0.0;

/* Variables B&B non utilisées en mode SDP-only, mais déclarées pour l'édition */
double ub_bb   = 1e10;
double lb_bb   = -1e10;
double *best_x = NULL;
long   nodecount     = 0;
long   nodecount_bis = 0;

/* Buffers de noms de fichiers (requis par in_out.c / parameters.c) */
char *buf_in    = NULL;
char *buf_out   = NULL;
char *buf_ampl  = NULL;
char *buf_local_sol = NULL;

/* ---------------------------------------------------------------------- */
/* Helpers internes                                                         */
/* ---------------------------------------------------------------------- */

/* Copie un tableau Python double[n] vers un vecteur C alloué */
static double *_copy_vec(const double *src, int n)
{
    double *dst = alloc_vector_d(n);
    memcpy(dst, src, n * sizeof(double));
    return dst;
}

/* Copie un tableau Python double[n1*n2] (row-major) vers alloc_matrix_d(n1,n2)
 * La librairie stocke M[ij2k(i,j,n2)] = M[i*n2+j], identique au row-major. */
static double *_copy_matrix(const double *src, int n1, int n2)
{
    double *dst = alloc_matrix_d(n1, n2);
    memcpy(dst, src, (size_t)n1 * n2 * sizeof(double));
    return dst;
}

/* Copie un tenseur 3D Python double[n1*n2*n2] (row-major) vers alloc_matrix_3d(n1,n2,n2).
 * ijk2l(i,j,k,n2,n2) = i*n2*n2 + j*n2 + k, identique au row-major numpy. */
static double *_copy_matrix_3d(const double *src, int n1, int n2)
{
    double *dst = alloc_matrix_3d(n1, n2, n2);
    memcpy(dst, src, (size_t)n1 * n2 * n2 * sizeof(double));
    return dst;
}

/* ---------------------------------------------------------------------- */
/* API publique                                                             */
/* ---------------------------------------------------------------------- */

/**
 * miqcr_populate_qp — remplit le struct qp (MIQCP) depuis les tableaux Python.
 *
 * Tableaux attendus (contiguous double, row-major) :
 *   Q   : n×n    matrice hessienne de l'objectif (symétrique)
 *   c   : n      partie linéaire
 *   u   : n      bornes supérieures des variables
 *   l   : n      bornes inférieures
 *   A   : m×n    (peut être NULL/vide si m=0) égalités linéaires
 *   b   : m      second membre des égalités
 *   D   : p×n    (peut être NULL/vide si p=0) inégalités linéaires ≤
 *   e   : p
 *   Aq  : mq×(n+1)×(n+1)  (peut être NULL/vide si mq=0) quadratiques =
 *   bq  : mq
 *   Dq  : pq×(n+1)×(n+1)  (peut être NULL/vide si pq=0) quadratiques ≤
 *   eq  : pq
 *   cons : constante de l'objectif
 */
void miqcr_populate_qp(
    int n, int nb_int,
    int m, int p,
    int mq, int pq,
    const double *Q,  const double *c,
    const double *u,  const double *l,
    const double *A,  const double *b,
    const double *D,  const double *e,
    const double *Aq, const double *bq,
    const double *Dq, const double *eq,
    double cons)
{
    /* Libère l'éventuel qp précédent */
    if (qp != NULL) {
        free_miqcp(qp);
        free(qp);
        qp = NULL;
    }

    qp = new_miqcp();
    if (qp == NULL) {
        fprintf(stderr, "[miqcr_pyapi] Échec d'allocation de qp\n");
        return;
    }

    /* Dimensions */
    qp->n      = n;
    qp->nb_int = nb_int;
    qp->m      = m;
    qp->p      = p;
    qp->mq     = mq;
    qp->pq     = pq;
    qp->cons   = cons;
    qp->flag_const = (fabs(cons) > 1e-15) ? 1 : 0;
    qp->miqp   = (nb_int < n) ? 1 : 0;  /* contient des cont. si nb_int < n */

    /* Objectif */
    qp->q = _copy_matrix(Q, n, n);
    qp->c = _copy_vec(c, n);

    /* Bornes (u,l sont de taille n + n*(n+1)/2 dans la lib originale,
     * mais pour la phase SDP seules les n premières entrées sont utilisées.
     * On alloue n + n*(n+1)/2 pour compatibilité et initialise le reste à 0). */
    {
        int N = n + n * (n + 1) / 2;
        qp->u = alloc_vector_d(N);
        qp->l = alloc_vector_d(N);
        memcpy(qp->u, u, n * sizeof(double));
        memcpy(qp->l, l, n * sizeof(double));
        for (int i = n; i < N; i++) {
            qp->u[i] = 1e30;
            qp->l[i] = -1e30;
        }
        qp->init_u = _copy_vec(qp->u, N);
        qp->init_l = _copy_vec(qp->l, N);
    }
    qp->N = n + n * (n + 1) / 2;

    /* Contraintes linéaires égalité : A x = b */
    if (m > 0 && A != NULL) {
        qp->a = _copy_matrix(A, m, n);
        qp->b = _copy_vec(b, m);
    } else {
        qp->a = alloc_matrix_d(1, 1);
        qp->b = alloc_vector_d(1);
    }

    /* Contraintes linéaires inégalité : D x ≤ e */
    if (p > 0 && D != NULL) {
        qp->d = _copy_matrix(D, p, n);
        qp->e = _copy_vec(e, p);
    } else {
        qp->d = alloc_matrix_d(1, 1);
        qp->e = alloc_vector_d(1);
    }

    /* Contraintes quadratiques égalité : x^T Aq_k x = bq_k
     * Aq est de taille mq × (n+1) × (n+1), identique au format ijk2l de la lib. */
    if (mq > 0 && Aq != NULL) {
        qp->aq = _copy_matrix_3d(Aq, mq, n + 1);
        qp->bq = _copy_vec(bq, mq);
    } else {
        qp->aq = alloc_matrix_3d(1, 1, 1);
        qp->bq = alloc_vector_d(1);
    }

    /* Contraintes quadratiques inégalité : x^T Dq_k x ≤ eq_k */
    if (pq > 0 && Dq != NULL) {
        qp->dq = _copy_matrix_3d(Dq, pq, n + 1);
        qp->eq = _copy_vec(eq, pq);
    } else {
        qp->dq = alloc_matrix_3d(1, 1, 1);
        qp->eq = alloc_vector_d(1);
    }

    /* Champs solution (non utilisés ici, initialisés à NULL/0) */
    qp->sol      = NULL;
    qp->sol_adm  = 1e30;
    qp->local_sol = NULL;

    /* Construit la structure SDP interne à partir de qp */
    create_sdp_mixed();

    printf("[miqcr_pyapi] qp peuplé : n=%d nb_int=%d m=%d p=%d mq=%d pq=%d\n",
           n, nb_int, m, p, mq, pq);
}


/**
 * miqcr_compute_betas — exécute la Conic Bundle et récupère les betas.
 *
 * Doit être appelée après miqcr_populate_qp().
 *
 * Paramètres de sortie (tableaux pré-alloués par Python) :
 *   out_beta      : double[n*n]  — betas finaux β_{ij} (row-major)
 *   out_alphaq    : double[mq]   — duaux Aq
 *   out_alphabisq : double[pq]   — duaux Dq
 *   out_sol_sdp   : double[1]    — valeur SDP à convergence
 */
void miqcr_compute_betas(
    double *out_beta,
    double *out_alphaq,
    double *out_alphabisq,
    double *out_sol_sdp)
{
    if (qp == NULL) {
        fprintf(stderr, "[miqcr_pyapi] Erreur : miqcr_populate_qp() n'a pas été appelé\n");
        return;
    }

    int n  = qp->n;
    int mq = qp->mq;
    int pq = qp->pq;

    /* Lance la phase SDP : Conic Bundle + SDP interne */
    compute_alpha_beta_sbb_miqcr();

    /* Copie les betas finaux (matrice n×n, row-major = ij2k) */
    if (psdp.beta != NULL) {
        memcpy(out_beta, psdp.beta, (size_t)n * n * sizeof(double));
    } else {
        memset(out_beta, 0, (size_t)n * n * sizeof(double));
    }

    /* Duaux des contraintes quadratiques */
    if (mq > 0 && psdp.alphaq != NULL) {
        memcpy(out_alphaq, psdp.alphaq, (size_t)mq * sizeof(double));
    }
    if (pq > 0 && psdp.alphabisq != NULL) {
        memcpy(out_alphabisq, psdp.alphabisq, (size_t)pq * sizeof(double));
    }

    /* Valeur SDP */
    *out_sol_sdp = sol_sdp;

    printf("[miqcr_pyapi] Phase SDP terminée : sol_sdp = %g\n", sol_sdp);
}


/**
 * miqcr_set_param — surcharge un paramètre numérique de parameters.h.
 *
 * Noms supportés (insensible à la casse) :
 *   "FACTOR", "EPS_TERM_CB", "NB_MAX_ITER", "TIME_LIMIT_CB",
 *   "EVAL_LIMIT", "UPDATE_LIMIT", "EPS_BETA", "EPS_SDP",
 *   "ACT_BOUND", "NB_DESCENT_STEP", "EPS_STEP"
 */
void miqcr_set_param(const char *name, double value)
{
    if      (strcasecmp(name, "FACTOR")          == 0) FACTOR          = value;
    else if (strcasecmp(name, "EPS_TERM_CB")     == 0) EPS_TERM_CB     = value;
    else if (strcasecmp(name, "NB_MAX_ITER")     == 0) NB_MAX_ITER     = (int)value;
    else if (strcasecmp(name, "TIME_LIMIT_CB")   == 0) TIME_LIMIT_CB   = (int)value;
    else if (strcasecmp(name, "EVAL_LIMIT")      == 0) EVAL_LIMIT      = (int)value;
    else if (strcasecmp(name, "UPDATE_LIMIT")    == 0) UPDATE_LIMIT    = (int)value;
    else if (strcasecmp(name, "EPS_BETA")        == 0) EPS_BETA        = value;
    else if (strcasecmp(name, "EPS_SDP")         == 0) EPS_SDP         = value;
    else if (strcasecmp(name, "ACT_BOUND")       == 0) ACT_BOUND       = (int)value;
    else if (strcasecmp(name, "NB_DESCENT_STEP") == 0) NB_DESCENT_STEP = (int)value;
    else if (strcasecmp(name, "EPS_STEP")        == 0) EPS_STEP        = value;
    else if (strcasecmp(name, "TRIANGLE")        == 0) TRIANGLE        = (int)value;
    else if (strcasecmp(name, "TRIANGLEB")       == 0) TRIANGLEB       = (int)value;
    else
        fprintf(stderr, "[miqcr_pyapi] Paramètre inconnu : %s\n", name);

    printf("[miqcr_pyapi] Paramètre %s = %g\n", name, value);
}
