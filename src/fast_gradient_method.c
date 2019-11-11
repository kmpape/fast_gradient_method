#include <assert.h>
#include <stdlib.h>
#include <math.h>

#include "fast_gradient_method.h"

#if defined(FGM_DEBUG) || defined(FGM_PROFILING)
#include <stdio.h>
#endif
#if defined(FGM_PROFILING)
#include <time.h>
#endif

/* Algorithm:
 *
 * for ()
 *   t(i) = (I - J / L) y(i) - q / L
 *   z(i+1) = P(t(i))
 *   y(i+1) = (1 + beta) z(i+1) - beta z(i)
 * end
 *
 * (I - J / L)  = FGM_in_mat
 * q / L        = FGM_in_vec
 * y(i), y(i+1) = out
 * z(i+1)       = FGM_vec_z_new
 * z(i)         = FGM_vec_z_old
 * t(i)         = FGM_vec_t
 *
 */

/* Problem Data */
#ifdef FGM_EMBEDDED
fgm_float FGM_in_mat[FGM_DIM * FGM_DIM];
fgm_float FGM_in_vec[FGM_DIM];
fgm_float FGM_vec_t[FGM_DIM];
fgm_float FGM_vec_z_new[FGM_DIM];
fgm_float FGM_vec_z_old[FGM_DIM];
#else
fgm_float * FGM_in_mat;
fgm_float * FGM_in_vec;
fgm_float * FGM_vec_t;
fgm_float * FGM_vec_z_new;
fgm_float * FGM_vec_z_old;
int FGM_DIM;
#endif
fgm_float FGM_max_eigval;
fgm_float FGM_min_eigval;
fgm_float FGM_beta;
fgm_float FGM_beta_p1;
int FGM_last_num_iter;
int FGM_is_initialized;
void (*FGM_project) (const fgm_float * restrict in, fgm_float * restrict out);
// we also make use of the out-array passed to solve()

/* Prototypes */
//void FGM_gradient_step(const fgm_float * restrict in, fgm_float * restrict out);
//void FGM_project(const fgm_float * restrict in, fgm_float * restrict out);

/* Prototypes algebra */
void FGM_vec_copy(const fgm_float * restrict in, fgm_float * restrict out, const int len, const double scaling_factor);
void FGM_vec_swap(fgm_float **in_out1, fgm_float **in_out2);
void FGM_vec_init(fgm_float * out, const fgm_float in);
fgm_float FGM_max(fgm_float in1, fgm_float in2);
fgm_float FGM_abs_float(fgm_float in);
fgm_float FGM_inf_norm(const fgm_float * in);
fgm_float FGM_inf_norm_error(const fgm_float * in1, const fgm_float * in2);
void FGM_beta_step(const fgm_float * restrict in1, const fgm_float * restrict in2, fgm_float * restrict out);

/* Profiling Utilities */
#ifdef FGM_PROFILING
void FGM_init_timers(void);
void FGM_tic(const int i_timer);
void FGM_toc(const int i_timer);
FGM_Timer FGM_total_timer;
#if (FGM_PROFILING_LEVEL == 2)
FGM_Timer FGM_grad_step_timer;
FGM_Timer FGM_proj_step_timer;
FGM_Timer FGM_beta_step_timer;
FGM_Timer FGM_term_step_timer;
FGM_Timer FGM_all_timers[4];
char * FGM_timer_names[5] = {"total", "grad_step", "proj_step", "beta_step", "term_step"};
const int FGM_n_timers = 5;
#else
FGM_Timer FGM_all_timers[1];
char * FGM_timer_names[5] = {"total"};
const int FGM_n_timers = 1;
#endif /* FGM_PROFILING_LEVEL */
#endif /* FGM_PROFILING */

int FGM_solve(fgm_float * out, const int warm_start) {
	/* Algorithm:
	 *
	 * for ()
	 *   t(i) = (I - J / L) y(i) - q / L
	 *   z(i+1) = P(t(i))
	 *   y(i+1) = (1 + beta) z(i+1) - beta z(i)
	 * end
	 *
	 * (I - J / L)  = FGM_in_mat
	 * q / L        = FGM_in_vec
	 * y(i), y(i+1) = out
	 * z(i+1)       = FGM_vec_z_new
	 * z(i)         = FGM_vec_z_old
	 * t(i)         = FGM_vec_t
	 *
	 */
	int i_iter;
	fgm_float abs_error, last_iter_inf_norm;
	assert(FGM_is_initialized == 1);

	if (warm_start != 1)
		FGM_vec_init(out, 0.0);

#ifdef FGM_PROFILING
	FGM_init_timers();
	FGM_tic(0);
#endif
	for (i_iter = 0; i_iter < FGM_MAX_ITER; i_iter ++) {
#ifdef FGM_DEBUG
		if (i_iter <= 2) {
			printf("i_iter=%d +++++++\n", i_iter);
		}
#endif /* FGM_DEBUG */
		/*
		 * FGM_vec_t = (I - J / L) out - FGM_in_vec
		 */

#if defined(FGM_PROFILING) && (FGM_PROFILING_LEVEL > 1)
		FGM_tic(1);
#endif
		FGM_gradient_step(out, FGM_vec_t);
#if defined(FGM_PROFILING) && (FGM_PROFILING_LEVEL > 1)
		FGM_toc(1);
#endif
#ifdef FGM_DEBUG
		if (i_iter <= 2) {
			int i;
			printf("Gradient Step:\n");
			for (i = 0; i < FGM_DIM; i++)
				printf("FGM_vec_t[%d]=%.4f\n", i, FGM_vec_t[i]);
		}
#endif /* FGM_DEBUG */

		/*
		 * z(i+1) = P(t(i))
		 */
#if defined(FGM_PROFILING) && (FGM_PROFILING_LEVEL > 1)
		FGM_tic(2);
#endif
		FGM_vec_swap(&FGM_vec_z_new, &FGM_vec_z_old);
		FGM_project(FGM_vec_t, FGM_vec_z_new);
#if defined(FGM_PROFILING) && (FGM_PROFILING_LEVEL > 1)
		FGM_toc(2);
#endif
#ifdef FGM_DEBUG
		if (i_iter <= 2) {
			int i;
			printf("Projection Step:\n");
			for (i = 0; i < FGM_DIM; i++)
				printf("FGM_vec_z_new[%d]=%.4f\n", i, FGM_vec_z_new[i]);
		}
#endif /* FGM_DEBUG */

		/*
		 * y(i+1) = (1 + beta) z(i+1) - beta z(i)
		 */
#if defined(FGM_PROFILING) && (FGM_PROFILING_LEVEL > 1)
		FGM_tic(3);
#endif
		FGM_beta_step(FGM_vec_z_new, FGM_vec_z_old, out);
#if defined(FGM_PROFILING) && (FGM_PROFILING_LEVEL > 1)
		FGM_toc(3);
#endif
#ifdef FGM_DEBUG
		if (i_iter <= 2) {
			int i;
			printf("Beta Step (beta=%.6f, beta_p1=%.6f):\n", FGM_beta, FGM_beta_p1);
			for (i = 0; i < FGM_DIM; i++)
				printf("out[%d]=%.4f\n", i, out[i]);
		}
#endif /* FGM_DEBUG */

		/*
		 * Check for termination.
		 */
#if defined(FGM_PROFILING) && (FGM_PROFILING_LEVEL > 1)
		FGM_tic(4);
#endif
		if ((FGM_CHECK_TERMINATION) && (i_iter > 0) && (i_iter % FGM_CHECK_TERMINATION == 0)) {
			abs_error = FGM_inf_norm_error(FGM_vec_z_new, FGM_vec_z_old);
			if (abs_error == 0) {
				break;
			} else {
				last_iter_inf_norm = FGM_inf_norm(FGM_vec_z_old);
				if (last_iter_inf_norm == 0) {
					break;
				} else {
					if ((abs_error < FGM_EPS_ABS) && (abs_error < FGM_EPS_REL * last_iter_inf_norm)) {
						break;
					}
				}
			}
		}
#if defined(FGM_PROFILING) && (FGM_PROFILING_LEVEL > 1)
		FGM_toc(4);
#endif
	}
#if defined(FGM_PROFILING)
	FGM_toc(0);
#endif
	FGM_last_num_iter = i_iter;
	if (i_iter == FGM_MAX_ITER) {
		return 1;
	} else {
		return 0;
	}
}

void FGM_gradient_step(const fgm_float * restrict in, fgm_float * restrict out) {
	int i_row, i_col;
	fgm_float row_res;
	fgm_float * mat_ptr;
	for (i_row = 0; i_row < FGM_DIM; i_row++) {
		row_res = 0.0;
		mat_ptr = &FGM_in_mat[i_row * FGM_DIM];
		for (i_col = 0; i_col < FGM_DIM; i_col++, mat_ptr++) {
			row_res += (*mat_ptr) * in[i_col];
		}
		out[i_row] = row_res - FGM_in_vec[i_row];
	}
}

void FGM_beta_step(const fgm_float * restrict in1, const fgm_float * restrict in2, fgm_float * restrict out) {
	int i_row;
	for (i_row = 0; i_row < FGM_DIM; i_row++) {
		out[i_row] = FGM_beta_p1 * in1[i_row] - FGM_beta * in2[i_row];
	}
}

/*
 * Algebra and miscellaneous
 */
void FGM_initialize(const fgm_float * obj_func_matrix, const fgm_float * obj_func_vector,
					const fgm_float obj_func_grad_max_eigval, const fgm_float obj_func_grad_min_eigval,
					const int fgm_dim, void (*proj_func) (const fgm_float * restrict in, fgm_float * restrict out)) {
	int i, j;
	const double sqrt_max_eigval = sqrt(obj_func_grad_max_eigval);
	const double sqrt_min_eigval = sqrt(obj_func_grad_min_eigval);
#ifndef FGM_EMBEDDED
	FGM_DIM = fgm_dim;
	FGM_in_mat = (fgm_float *)malloc(FGM_DIM * FGM_DIM * sizeof(fgm_float));
	FGM_in_vec = (fgm_float *)malloc(FGM_DIM * sizeof(fgm_float));
	FGM_vec_t = (fgm_float *)malloc(FGM_DIM * sizeof(fgm_float));
	FGM_vec_z_new = (fgm_float *)malloc(FGM_DIM * sizeof(fgm_float));
	FGM_vec_z_old = (fgm_float *)malloc(FGM_DIM * sizeof(fgm_float));
#else
	assert(fgm_dim == FGM_DIM);
#endif
	assert(obj_func_grad_max_eigval > obj_func_grad_min_eigval);

	FGM_beta = (fgm_float) (sqrt_max_eigval - sqrt_min_eigval) / (sqrt_max_eigval + sqrt_min_eigval);
	FGM_beta_p1 = (fgm_float) ((sqrt_max_eigval - sqrt_min_eigval) / (sqrt_max_eigval + sqrt_min_eigval) + 1.0);

	// For the vector, directly compute:  q / max_eigval
	for (i = 0; i < FGM_DIM; i++) {
		FGM_in_vec[i] = (fgm_float)(((double)obj_func_vector[i]) / (double)obj_func_grad_max_eigval);
	}

	// For the matrix, directly compute: eye(FGM_DIM,FGM_DIM) - J / max_eigval
	for (i = 0; i < FGM_DIM; i++) {
		for (j = 0; j < FGM_DIM; j++) {
			if (i == j) {
				FGM_in_mat[i * FGM_DIM + j] = (fgm_float)(1.0 - (double)obj_func_matrix[i * FGM_DIM + j] / (double)obj_func_grad_max_eigval);
			} else {
				FGM_in_mat[i * FGM_DIM + j] = (fgm_float)(- (double)obj_func_matrix[i * FGM_DIM + j] / (double)obj_func_grad_max_eigval);
			}
		}
	}

	FGM_project = proj_func;
	FGM_is_initialized = 1;
}

void FGM_finalize(void) {
#ifndef FGM_EMBEDDED
	free(FGM_in_mat);
	free(FGM_in_vec);
	free(FGM_vec_t);
	free(FGM_vec_z_new);
	free(FGM_vec_z_old);
#endif
}

void FGM_vec_init(fgm_float * out, const fgm_float in) {
	int i;
	for (i = 0; i < FGM_DIM; i++)
		out[i] = in;
}

fgm_float FGM_compute_obj_val(fgm_float * solution, fgm_float * obj_fun_matrix, fgm_float * obj_fun_vec) {
	int i_row, i_col;
	fgm_float matrix_part = 0.0, vector_part = 0.0, tmp_res;

	for (i_row = 0; i_row < FGM_DIM; i_row++) {
		tmp_res = 0.0;
		for (i_col = 0; i_col < FGM_DIM; i_col++) {
			tmp_res += obj_fun_matrix[i_row * FGM_DIM + i_col] * solution[i_col];
		}
		matrix_part += solution[i_row] * tmp_res;
		vector_part += solution[i_row] * obj_fun_vec[i_row];
	}

	return (0.5 * matrix_part + vector_part);
}

/*
void swap_vectors(c_float **a, c_float **b) {
  c_float *temp;

  temp = *b;
  *b   = *a;
  *a   = temp;
}
*/
void FGM_vec_swap(fgm_float **in_out1, fgm_float **in_out2) {
	fgm_float * tmp;
	tmp = *in_out2;
	*in_out2 = *in_out1;
	*in_out1 = tmp;
}

void FGM_vec_copy(const fgm_float * restrict in, fgm_float * restrict out, const int len, const double scaling_factor) {
	int i;
	for (i = 0; i < len; i++) {
		out[i] = (fgm_float)(((double)in[i]) * scaling_factor);
	}
}

fgm_float FGM_max(fgm_float in1, fgm_float in2) {
	return (in1 > in2) ? in1 : in2;
}

fgm_float FGM_abs_float(fgm_float in) {
	return (in > 0) ? in : -in;
}

fgm_float FGM_inf_norm(const fgm_float * in) {
	int i;
	fgm_float max_val = FGM_abs_float(in[0]);
	for (i = 1; i < FGM_DIM; i++)
		max_val = FGM_max(FGM_abs_float(in[i]), max_val);
	return max_val;
}

fgm_float FGM_inf_norm_error(const fgm_float * in1, const fgm_float * in2) {
	int i;
	fgm_float max_val = FGM_abs_float(in1[0] - in2[0]);
	for (i = 1; i < FGM_DIM; i++)
		max_val = FGM_max(FGM_abs_float(in1[i] - in2[i]), max_val);
	return max_val;
}

int FGM_get_num_iter(void) {
	return FGM_last_num_iter;
}

#ifdef FGM_PROFILING
FGM_Timer * FGM_get_all_timers(void) {
	return FGM_all_timers;
}


int FGM_get_num_timers(void) {
	return FGM_n_timers;
}

char ** FGM_get_timer_names(void) {
	return FGM_timer_names;
}

void FGM_init_timers(void) {
	int i;
	FGM_all_timers[0] = FGM_total_timer;
#if (FGM_PROFILING_LEVEL > 1)
	FGM_all_timers[1] = FGM_grad_step_timer;
	FGM_all_timers[2] = FGM_proj_step_timer;
	FGM_all_timers[3] = FGM_beta_step_timer;
	FGM_all_timers[4] = FGM_term_step_timer;
#endif
	for (i = 0; i < FGM_n_timers; i++) {
		FGM_all_timers[i].t_cumsum = 0.0;
		FGM_all_timers[i].n_calls = 0;
	}
}

void FGM_tic(const int i_timer) {
	clock_gettime(CLOCK_MONOTONIC, &(FGM_all_timers[i_timer].tic));
}

void FGM_toc(const int i_timer) {
	struct timespec temp;

	clock_gettime(CLOCK_MONOTONIC, &(FGM_all_timers[i_timer].toc));

	if ((FGM_all_timers[i_timer].toc.tv_nsec - FGM_all_timers[i_timer].tic.tv_nsec) < 0) {
		temp.tv_sec  = FGM_all_timers[i_timer].toc.tv_sec - FGM_all_timers[i_timer].tic.tv_sec - 1;
		temp.tv_nsec = 1e9 + FGM_all_timers[i_timer].toc.tv_nsec - FGM_all_timers[i_timer].tic.tv_nsec;
	} else {
		temp.tv_sec  = FGM_all_timers[i_timer].toc.tv_sec - FGM_all_timers[i_timer].tic.tv_sec;
		temp.tv_nsec = FGM_all_timers[i_timer].toc.tv_nsec - FGM_all_timers[i_timer].tic.tv_nsec;
	}

	FGM_all_timers[i_timer].t_cumsum += (float)temp.tv_sec + (float)temp.tv_nsec / 1e9;
	FGM_all_timers[i_timer].n_calls += 1;
}

float FGM_average_time(FGM_Timer *t) {
	return t->t_cumsum / ((float)t->n_calls);
}

float FGM_total_time(FGM_Timer *t) {
	return t->t_cumsum;
}

void FGM_print_timer_info(FGM_Timer timer, char *timer_name) {
	const double avg_time_seconds = FGM_average_time(&timer);
	const double total_time_seconds = FGM_total_time(&timer);
	const long int ncycles = 0;
	printf("%s: Total time = %.6f s e-6, Time / n_calls = %.6f s e-6 (n_calls=%d) (Freq = %.6f Hz) (Cycles = %ld)\n",
			timer_name, total_time_seconds*1000000.0, avg_time_seconds*1000000.0, (int)timer.n_calls,
			1.0 / (avg_time_seconds), ncycles);
}
#endif
