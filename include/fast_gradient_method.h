
#ifndef FAST_GRADIENT_METHOD_H_
#define FAST_GRADIENT_METHOD_H_

/* Parameters */
#define DOUBLE_PRECISION
#ifdef DOUBLE_PRECISION
typedef double fgm_float;
#define FGM_EPS_ABS (1e-4)
#define FGM_EPS_REL (1e-4)
#else
typedef float fgm_float;
#define FGM_EPS_ABS (1e-4f)
#define FGM_EPS_REL (1e-4f)
#endif

#ifdef FGM_EMBEDDED
#define FGM_DIM (10)
#endif


#define FGM_MAX_ITER (1000)
#define FGM_CHECK_TERMINATION (10)

/* Functions */

/*
 * FGM_initialize:
 * Copies problem data and assigns the projection function: void proj_func(const float * restrict in, float * restrict out).
 * Needs to be called before FGM_solve.
 */
void FGM_initialize(const fgm_float * obj_func_matrix, const fgm_float * obj_func_vector,
		const fgm_float obj_func_grad_max_eigval, const fgm_float obj_func_grad_min_eigval,
		const int fgm_dim, void (*proj_func) (const fgm_float * restrict in, fgm_float * restrict out));
void FGM_finalize(void);

/*
 * FGM_solve:
 * Solves the QP. Warm-start using out-array if warm_start == 1. Solution in out. Returns 0
 * if problem has been solved. Returns 1 if maximum iterations reached.
 */
int FGM_solve(fgm_float * out, const int warm_start);
int FGM_get_num_iter(void);
fgm_float FGM_compute_obj_val(fgm_float * solution, fgm_float * obj_fun_matrix, fgm_float * obj_fun_vec);

/* Only here for tests */
void FGM_gradient_step(const fgm_float * restrict in, fgm_float * restrict out);

/* Profiling */
#define FGM_PROFILING // enable profiling
#define FGM_PROFILING_LEVEL (1) // 2 for detailed timing, 1 for total (loop) timing only

#ifdef FGM_PROFILING
typedef struct FGMTimer {
#ifdef SOC_C6678
	unsigned long int tic;
	unsigned long int toc;
#else
	struct timespec tic;
	struct timespec toc;
#endif
	int n_calls;
	float t_cumsum;
} FGM_Timer;
int FGM_get_num_timers(void);
FGM_Timer * FGM_get_all_timers(void); // returns 5 timers for FGM_PROFILING_LEVEL==2,  1 timer else
char ** FGM_get_timer_names(void);
float FGM_average_time(FGM_Timer *t);
float FGM_total_time(FGM_Timer *t);
void FGM_print_timer_info(FGM_Timer timer, char *timer_name);
#endif

#endif /* FAST_GRADIENT_METHOD_H_ */
