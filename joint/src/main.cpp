#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <gsl/gsl_rng.h>
#include "linelib.h"

char nodes_file[MAX_STRING], words_file[MAX_STRING], hin_file[MAX_STRING], output_file[MAX_STRING], type_file[MAX_STRING];
int binary = 0, num_threads = 1, vector_size = 100, negative = 5, edge_type_count = 5;
long long samples = 1, edge_count_actual;
real alpha = 0.025, margin = 0.3, starting_alpha;

const gsl_rng_type * gsl_T;
gsl_rng * gsl_r;

line_node nodes, words;
line_hin text_hin;
std::vector<line_trainer> trainer_array;

double func_rand_num()
{
	return gsl_rng_uniform(gsl_r);
}

void *TrainModelThread(void *id) 
{
	long long edge_count = 0, last_edge_count = 0;
	unsigned long long next_random = (long long)id;
	real obj = 0;

	while (1)
	{
		if (edge_count > samples / num_threads + 2) break;

		if (edge_count - last_edge_count > 10000)
		{
			edge_count_actual += edge_count - last_edge_count;
			last_edge_count = edge_count;
			printf("%cAlpha: %f Objective: %f Progress: %.3lf%%", 13, alpha, obj, (real)edge_count_actual / (real)(samples + 1) * 100);
			fflush(stdout);
			alpha = starting_alpha * (1 - edge_count_actual / (real)(samples + 1));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
			obj = 0;
		}

		for (int i = 0; i < (int)trainer_array.size(); i++)
			obj += trainer_array[i].train_sample(alpha, margin, func_rand_num, next_random);
		
		edge_count += trainer_array.size();
	}
	pthread_exit(NULL);
}

// void *TrainModelThread(void *id) 
// {
// 	long long edge_count = 0, last_edge_count = 0;
// 	unsigned long long next_random = (long long)id;
// 	real *error_vec = (real *)calloc(vector_size, sizeof(real));

// 	while (1)
// 	{
// 		if (edge_count > samples / num_threads + 2) break;

// 		if (edge_count - last_edge_count>10000)
// 		{
// 			edge_count_actual += edge_count - last_edge_count;
// 			last_edge_count = edge_count;
// 			printf("%cAlpha: %f Progress: %.3lf%%", 13, alpha, (real)edge_count_actual / (real)(samples + 1) * 100);
// 			fflush(stdout);
// 			alpha = starting_alpha * (1 - edge_count_actual / (real)(samples + 1));
// 			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
// 		}

// 		for (int i = 0; i < (int)trainer_array.size(); i++)
// 			trainer_array[i].train_sample(alpha, error_vec, func_rand_num, next_random);

// 		edge_count += 3;
// 	}
// 	free(error_vec);
// 	pthread_exit(NULL);
// }

void TrainModel() {
	long a;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	starting_alpha = alpha;

	nodes.init(nodes_file, vector_size);
	words.init(words_file, vector_size);
	text_hin.init(hin_file, &words, &nodes);

	for (int i = 0; i < (int)trainer_array.size(); i++) trainer_array[i].init(i, &text_hin, negative);

	clock_t start = clock();
	printf("Training process:\n");
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	printf("\n");
	clock_t finish = clock();
	printf("Total time: %lf\n", (double)(finish - start) / CLOCKS_PER_SEC);

	// words.output(output_file, binary);
	nodes.output(output_file, binary);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	output_file[0] = 0;
	if ((i = ArgPos((char *)"-words", argc, argv)) > 0) strcpy(words_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-nodes", argc, argv)) > 0) strcpy(nodes_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-hin", argc, argv)) > 0) strcpy(hin_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-type", argc, argv)) > 0) edge_type_count = atoi(argv[i + 1]); //strcpy(type_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) vector_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-samples", argc, argv)) > 0) samples = atoi(argv[i + 1])*(long long)(1000000);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);

	gsl_rng_env_setup();
	gsl_T = gsl_rng_rand48;
	gsl_r = gsl_rng_alloc(gsl_T);
	gsl_rng_set(gsl_r, 314159265);

	trainer_array.resize(edge_type_count);
	TrainModel();
	return 0;
}