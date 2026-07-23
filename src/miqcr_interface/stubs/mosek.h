/*
 * Stub mosek.h — used only to compile MIQCR sources with -fPIC.
 * The real Mosek functions are never called at runtime because
 * --wrap=run_sdp_solver_mixed_mosek intercepts all calls.
 */
#ifndef MOSEK_H_STUB
#define MOSEK_H_STUB

#include <stddef.h>

typedef int        MSKint32t;
typedef long long  MSKint64t;
typedef double     MSKrealt;
typedef int        MSKbooleant;

typedef void *MSKenv_t;
typedef void *MSKtask_t;
typedef void *MSKuserhandle_t;

typedef enum {
    MSK_RES_OK             = 0,
    MSK_RES_ERR_UNKNOWN    = 1
} MSKrescodee;

typedef enum {
    MSK_SOL_STA_UNKNOWN            = 0,
    MSK_SOL_STA_OPTIMAL            = 1,
    MSK_SOL_STA_PRIM_INFEAS_CER   = 2,
    MSK_SOL_STA_DUAL_INFEAS_CER   = 3
} MSKsolstae;

typedef enum {
    MSK_STREAM_LOG = 0,
    MSK_STREAM_MSG = 1
} MSKstreamtypee;

typedef enum {
    MSK_SOL_ITR = 0,
    MSK_SOL_BAS = 1
} MSKsoltypee;

typedef enum {
    MSK_OBJECTIVE_SENSE_MAXIMIZE = 1,
    MSK_OBJECTIVE_SENSE_MINIMIZE = 0
} MSKobjsensee;

typedef enum {
    MSK_BK_FR = 0,
    MSK_BK_LO = 1,
    MSK_BK_UP = 2,
    MSK_BK_FX = 3,
    MSK_BK_RA = 4
} MSKboundkeye;

#define MSK_INFINITY  1.0e30
#define MSK_MAX_STR_LEN 1024

#define MSKAPI

typedef void (MSKAPI *MSKstreamfunc)(void *handle, const char *str);

static inline MSKrescodee MSK_makeenv(MSKenv_t *env, const char *dbgfile)
    { (void)env; (void)dbgfile; return MSK_RES_OK; }
static inline MSKrescodee MSK_maketask(MSKenv_t env, MSKint32t maxnumcon, MSKint32t maxnumvar, MSKtask_t *task)
    { (void)env; (void)maxnumcon; (void)maxnumvar; (void)task; return MSK_RES_OK; }
static inline MSKrescodee MSK_deleteenv(MSKenv_t *env)
    { (void)env; return MSK_RES_OK; }
static inline MSKrescodee MSK_deletetask(MSKtask_t *task)
    { (void)task; return MSK_RES_OK; }
static inline void *MSK_calloctask(MSKtask_t task, size_t count, size_t size)
    { (void)task; return calloc(count, size); }

static inline MSKrescodee MSK_linkfunctotaskstream(MSKtask_t task, MSKstreamtypee whichstream, MSKuserhandle_t handle, MSKstreamfunc func)
    { (void)task; (void)whichstream; (void)handle; (void)func; return MSK_RES_OK; }
static inline MSKrescodee MSK_appendbarvars(MSKtask_t task, MSKint32t num, MSKint32t *dim)
    { (void)task; (void)num; (void)dim; return MSK_RES_OK; }
static inline MSKrescodee MSK_appendcons(MSKtask_t task, MSKint32t num)
    { (void)task; (void)num; return MSK_RES_OK; }
static inline MSKrescodee MSK_putconboundslice(MSKtask_t task, MSKint32t first, MSKint32t last, MSKboundkeye *bkc, MSKrealt *blc, MSKrealt *buc)
    { (void)task; (void)first; (void)last; (void)bkc; (void)blc; (void)buc; return MSK_RES_OK; }
static inline MSKrescodee MSK_putcfix(MSKtask_t task, MSKrealt cfix)
    { (void)task; (void)cfix; return MSK_RES_OK; }
static inline MSKrescodee MSK_putobjsense(MSKtask_t task, MSKobjsensee sense)
    { (void)task; (void)sense; return MSK_RES_OK; }
static inline MSKrescodee MSK_putbarcblocktriplet(MSKtask_t task, MSKint64t num, MSKint32t *subj, MSKint32t *subk, MSKint32t *subl, MSKrealt *valjkl)
    { (void)task; (void)num; (void)subj; (void)subk; (void)subl; (void)valjkl; return MSK_RES_OK; }
static inline MSKrescodee MSK_putbarablocktriplet(MSKtask_t task, MSKint64t num, MSKint32t *subi, MSKint32t *subj, MSKint32t *subk, MSKint32t *subl, MSKrealt *valijkl)
    { (void)task; (void)num; (void)subi; (void)subj; (void)subk; (void)subl; (void)valijkl; return MSK_RES_OK; }
static inline MSKrescodee MSK_optimizetrm(MSKtask_t task, MSKrescodee *trmcode)
    { (void)task; (void)trmcode; return MSK_RES_OK; }
static inline MSKrescodee MSK_solutionsummary(MSKtask_t task, MSKstreamtypee whichstream)
    { (void)task; (void)whichstream; return MSK_RES_OK; }
static inline MSKrescodee MSK_getsolsta(MSKtask_t task, MSKsoltypee whichsol, MSKsolstae *solutionsta)
    { (void)task; (void)whichsol; (void)solutionsta; return MSK_RES_OK; }
static inline MSKrescodee MSK_getbarxj(MSKtask_t task, MSKsoltypee whichsol, MSKint32t j, MSKrealt *barxj)
    { (void)task; (void)whichsol; (void)j; (void)barxj; return MSK_RES_OK; }
static inline MSKrescodee MSK_getslc(MSKtask_t task, MSKsoltypee whichsol, MSKrealt *slc)
    { (void)task; (void)whichsol; (void)slc; return MSK_RES_OK; }
static inline MSKrescodee MSK_getsuc(MSKtask_t task, MSKsoltypee whichsol, MSKrealt *suc)
    { (void)task; (void)whichsol; (void)suc; return MSK_RES_OK; }
static inline MSKrescodee MSK_getprimalobj(MSKtask_t task, MSKsoltypee whichsol, MSKrealt *primalobj)
    { (void)task; (void)whichsol; (void)primalobj; return MSK_RES_OK; }
static inline MSKrescodee MSK_getcodedesc(MSKrescodee code, char *symname, char *str)
    { (void)code; (void)symname; (void)str; return MSK_RES_OK; }
static inline MSKrescodee MSK_writedata(MSKtask_t task, const char *filename)
    { (void)task; (void)filename; return MSK_RES_OK; }

#endif /* MOSEK_H_STUB */
