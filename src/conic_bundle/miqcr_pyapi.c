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

/* Globals normalement définis dans smiqp.c (exclus car il contient main()) */
Q_MIQCP  qqp  = NULL;  /* problème quadratique étendu (non utilisé dans la phase SDP) */
C_MIQCP  cqp  = NULL;  /* problème continu relaxé (non utilisé dans la phase SDP) */
int      nb_var_cont   = 0;
int      nb_var_int    = 0;
int      nb_var_01     = 0;
int      nb_cont_01    = 0;
int      nb_cont_sup_1 = 0;
long     start_time    = 0;
int      nb_generator  = 0;
int      nb_nodes      = 0;
int     *selection_order = NULL;

/* ---------------------------------------------------------------------- */
/* Solveur SDP externe (remplace Mosek si enregistré)                      */
/* ---------------------------------------------------------------------- */

/* Type identique à celui déclaré dans solver_sdp_mixed.c */
typedef void (*sdp_solver_cb_t)(
    int n, int mq, int pq,
    const double *q_beta,
    const double *c_beta,
    double        l_beta,
    double       *x_out,
    double       *beta_diag_out,
    double       *alphaq_out,
    double       *alphabisq_out,
    double       *sol_sdp_out
);

/* Définition du pointeur global (extern dans solver_sdp_mixed.c) */
sdp_solver_cb_t _sdp_solve_cb = NULL;

/* Compteur d'itérations Conic Bundle (incrémenté dans __wrap_run_sdp_solver_mixed_mosek) */
static int _cb_iter_count = 0;

/* Buffer global : copie du triangle inférieur de X* à la dernière itération CB */
static double *_last_x_buf = NULL;
static int     _last_x_len = 0;

/**
 * miqcr_register_sdp_solver — enregistre un callback Python comme solveur SDP.
 *
 * À appeler depuis Python (via ctypes) après miqcr_populate_qp() et avant
 * miqcr_compute_betas(). Une fois enregistré, run_sdp_solver_mixed_mosek()
 * appelle ce callback à chaque itération de la Conic Bundle au lieu de Mosek.
 *
 * Le callback reçoit :
 *   n, mq, pq          : dimensions
 *   q_beta[n*n]        : partie quadratique de C_beta (row-major)
 *   c_beta[n]          : partie linéaire de C_beta
 *   l_beta             : constante scalaire
 *
 * Le callback doit écrire dans :
 *   x_out[(n+1)(n+2)/2]: triangle inférieur de X (format Mosek)
 *   beta_diag_out[n]   : duaux des contraintes diagonales
 *   alphaq_out[mq]     : duaux de <Aq_k,X>=bq_k
 *   alphabisq_out[pq]  : duaux de <Dq_k,X><=eq_k
 *   sol_sdp_out[1]     : valeur objective <C_beta,X>
 *
 * Passer cb=NULL pour revenir à Mosek.
 */
void miqcr_register_sdp_solver(sdp_solver_cb_t cb)
{
    _sdp_solve_cb = cb;
    if (cb != NULL)
        printf("[miqcr_pyapi] Solveur SDP externe enregistré.\n");
    else
        printf("[miqcr_pyapi] Retour au solveur Mosek.\n");
}


/* ---------------------------------------------------------------------- */
/* Interception de run_sdp_solver_mixed_mosek via --wrap du linker         */
/*                                                                          */
/* Avec -fPIC, les appels à run_sdp_solver_mixed_mosek dans                */
/* solver_sdp_mixed.c génèrent des relocations R_X86_64_PLT32 même pour   */
/* un appel intra-fichier. --wrap=run_sdp_solver_mixed_mosek les intercept */
/* tous → _cb_iter_count comptabilise les itérations CB dans les deux cas  */
/* (solveur Mosek interne ou callback Python enregistré).                  */
/* ---------------------------------------------------------------------- */

/* Déclaration de la version originale Mosek (résolue par le linker) */
extern void __real_run_sdp_solver_mixed_mosek(double **x);

/* Déclarations des fonctions MIQCR nécessaires pour le chemin callback */
extern void dualized_objective_function(SDP psdp,
                                        double **q_beta,
                                        double **c_beta,
                                        double  *l_beta);

void __wrap_run_sdp_solver_mixed_mosek(double **x)
{
    _cb_iter_count++;
    if (_sdp_solve_cb == NULL) {
        /* Aucun solveur externe : délègue à Mosek sans modifier l'état CB */
        __real_run_sdp_solver_mixed_mosek(x);
        return;
    }

    /* Chemin callback Python */
    double *q_beta = NULL;
    double *c_beta = NULL;
    double  l_beta = 0.0;
    dualized_objective_function(psdp, &q_beta, &c_beta, &l_beta);

    int n   = psdp->n;
    int mq  = psdp->mq;
    int pq  = psdp->pq;

    double *beta_diag_buf  = alloc_vector_d(n);
    double *alphaq_buf     = alloc_vector_d(mq > 0 ? mq : 1);
    double *alphabisq_buf  = alloc_vector_d(pq > 0 ? pq : 1);
    double  sol_out        = 0.0;

    _sdp_solve_cb(n, mq, pq,
                  q_beta, c_beta, l_beta,
                  x[0],
                  beta_diag_buf,
                  alphaq_buf,
                  alphabisq_buf,
                  &sol_out);

    psdp->beta_diag  = beta_diag_buf;
    if (mq > 0) psdp->alphaq    = alphaq_buf;
    if (pq > 0) psdp->alphabisq = alphabisq_buf;
    sol_sdp = sol_out;

    free(q_beta);
    free(c_beta);

    /* Sauvegarde X* (triangle inférieur) — uniquement dans le chemin callback */
    {
        int len = (n + 1) * (n + 2) / 2;
        if (_last_x_len != len) {
            free(_last_x_buf);
            _last_x_buf = (double *)malloc((size_t)len * sizeof(double));
            _last_x_len = len;
        }
        if (_last_x_buf != NULL && x[0] != NULL) {
            memcpy(_last_x_buf, x[0], (size_t)len * sizeof(double));
        }
    }
}

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
        free_miqcp();
        qp = NULL;
    }

    qp = new_miqcp();
    if (qp == NULL) {
        fprintf(stderr, "[miqcr_pyapi] Échec d'allocation de qp\n");
        return;
    }

    /* new_miqcp() utilise malloc() (pas calloc()), donc les champs non affectés
     * ci-dessous restent non initialisés.  free_miqcp() appelle
     * free_vector(qp->lg) et free_vector(qp->lgc) : si ces pointeurs sont
     * garbage, cela provoque "free(): invalid pointer" à l'appel suivant.
     * On les initialise à NULL pour que free(NULL) soit un no-op légal. */
    qp->lg  = NULL;
    qp->lgc = NULL;
    _cb_iter_count = 0;
    _last_x_len    = 0;   /* invalide le buffer X* du run précédent */

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

    /* Calcul des compteurs de types de variables (normalement fait dans smiqp.c
     * après read_file — absents ici car smiqp.c contient main() et est exclu).
     * Ces globaux dimensionnent les tableaux de contraintes dans run_sdp_solver_mixed_mosek ;
     * des valeurs à 0 provoquent des débordements de tampon → corruption heap. */
    nb_var_int    = 0;
    nb_var_01     = 0;
    nb_cont_01    = 0;
    nb_cont_sup_1 = 0;
    for (int _i = 0; _i < nb_int; _i++) {
        if (qp->u[_i] != 1) nb_var_int++;
        else                 nb_var_01++;
    }
    for (int _i = nb_int; _i < n; _i++) {
        if (qp->u[_i] == 1 && qp->l[_i] == 0) nb_cont_01++;
        if (qp->l[_i] >= 1)                     nb_cont_sup_1++;
    }
    nb_var_cont = n - nb_var_int - nb_var_01;

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
    if (psdp->beta != NULL) {
        memcpy(out_beta, psdp->beta, (size_t)n * n * sizeof(double));
    } else {
        memset(out_beta, 0, (size_t)n * n * sizeof(double));
    }

    /* Duaux des contraintes quadratiques */
    if (mq > 0 && psdp->alphaq != NULL) {
        memcpy(out_alphaq, psdp->alphaq, (size_t)mq * sizeof(double));
    }
    if (pq > 0 && psdp->alphabisq != NULL) {
        memcpy(out_alphabisq, psdp->alphabisq, (size_t)pq * sizeof(double));
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


/**
 * miqcr_get_nb_iter — retourne le nombre d'itérations Conic Bundle de la dernière exécution.
 *
 * Chaque appel à run_sdp_solver_mixed_mosek (intercepté via --wrap) correspond
 * à une itération CB, que le solveur SDP soit Mosek ou Python.
 */
int miqcr_get_nb_iter(void)
{
    return _cb_iter_count;
}


/**
 * miqcr_get_final_x — copie le triangle inférieur de la matrice SDP finale X*
 * dans le buffer out_x (de taille (n+1)(n+2)/2, alloué par Python).
 *
 * À appeler après miqcr_compute_betas(). Le format est le triangle inférieur
 * row-major : out_x[r*(r+1)/2 + c] = X[r,c] pour r >= c (taille n+1).
 */
void miqcr_get_final_x(double *out_x)
{
    if (qp == NULL) {
        fprintf(stderr, "[miqcr_pyapi] miqcr_get_final_x: non initialisé\n");
        return;
    }
    int n   = qp->n;
    int len = (n + 1) * (n + 2) / 2;
    if (_last_x_buf != NULL && _last_x_len == len) {
        memcpy(out_x, _last_x_buf, (size_t)len * sizeof(double));
    } else {
        fprintf(stderr, "[miqcr_pyapi] miqcr_get_final_x: aucune solution SDP disponible\n");
        memset(out_x, 0, (size_t)len * sizeof(double));
    }
}


/**
 * miqcr_get_true_obj — évalue le vrai objectif (non pénalisé) sur X* final.
 *
 * MIQCR stocke psdp->q = -Q_orig et psdp->c = -c_orig (convention interne de
 * dualization, cf. dualized_objective_function dans utilities.c). La valeur
 * vraie est donc :
 *   true_obj = <Q_orig, X*[1:,1:]> + <c_orig, X*[1:,0]> + cons
 *            = -<psdp->q, X*[1:,1:]> - <psdp->c, X*[1:,0]> + psdp->cons
 *
 * make_matrix_with_vector() reconstruit X* depuis l'état interne de psdp
 * (valide après miqcr_compute_betas). Retourne 0.0 si non initialisé.
 */
extern void make_matrix_with_vector(SDP psdp, double **matrix);

double miqcr_get_true_obj(void)
{
    if (qp == NULL || psdp == NULL) {
        fprintf(stderr, "[miqcr_pyapi] miqcr_get_true_obj: non initialisé\n");
        return 0.0;
    }
    int n = qp->n;
    double *matrix = NULL;
    make_matrix_with_vector(psdp, &matrix);
    if (matrix == NULL) {
        fprintf(stderr, "[miqcr_pyapi] miqcr_get_true_obj: make_matrix_with_vector a retourné NULL\n");
        return 0.0;
    }

    double val = 0.0;
    int i, j;
    /* <Q_orig, X*[1:,1:]> = -<psdp->q, X*[1:,1:]> */
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            val -= psdp->q[ij2k(i, j, n)] * matrix[ij2k(i + 1, j + 1, n + 1)];
    /* <c_orig, X*[1:,0]> = -<psdp->c, X*[1:,0]> */
    for (i = 0; i < n; i++)
        val -= psdp->c[i] * matrix[ij2k(0, i + 1, n + 1)];
    /* + cons */
    val += psdp->cons;

    free_vector_d(matrix);
    printf("[miqcr_pyapi] miqcr_get_true_obj = %g\n", val);
    return val;
}
