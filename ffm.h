#ifndef _LIBFFM_H
#define _LIBFFM_H

#ifdef __cplusplus
extern "C" 
{

namespace ffm
{
#endif

typedef float ffm_float;
typedef double ffm_double;
typedef int ffm_int;
typedef long long ffm_long;

struct ffm_node
{
    ffm_int f;
    ffm_int j;
    ffm_float v;
};

struct ffm_problem
{
    ffm_int n;
    ffm_int l;
    ffm_int m;
    ffm_node *X;
    ffm_long *P;
    ffm_float *Y;
};

struct ffm_model
{
    ffm_int n;
    ffm_int m;
    ffm_int k;
    ffm_float *W;
    bool normalization;
};

struct ffm_parameter
{
    ffm_float eta;
    ffm_float lambda;
    ffm_int nr_iters;
    ffm_int k;
    ffm_int nr_threads;
    ffm_int negative_samples;
    ffm_int negative_position;
    bool quiet;
    bool normalization;
    bool random;
    bool auto_stop;
};

struct ffm_block_structure
{
    ffm_int max_feature;
    ffm_int max_field;
    ffm_int nr_features;
    ffm_int *index;
    ffm_node *features;
};

struct ffm_negative_sampling
{
    ffm_int negative_position;
    ffm_int num_negative_samples;
    ffm_int num_sampling_buckets;
    ffm_int *sampling_buckets;
};

ffm_problem* ffm_read_problem(char const *path);

int ffm_read_problem_to_disk(ffm_block_structure *bs, char const *txt_path, char const *bin_path);

void ffm_destroy_problem(struct ffm_problem **prob);

ffm_int ffm_save_model(ffm_model *model, char const *path);

ffm_model* ffm_load_model(char const *path);

void ffm_destroy_model(struct ffm_model **model);

ffm_parameter ffm_get_default_param();

ffm_model* ffm_train_with_validation(struct ffm_problem *Tr, struct ffm_problem *Va, struct ffm_negative_sampling *ns, struct ffm_block_structure *bs, struct ffm_parameter param);

ffm_model* ffm_train(struct ffm_problem *prob, struct ffm_negative_sampling *ns, struct ffm_block_structure *bs, struct ffm_parameter param);

ffm_model* ffm_train_with_validation_on_disk(char const *Tr_path, char const *Va_path, ffm_block_structure *bs, struct ffm_parameter param);

ffm_model* ffm_train_on_disk(char const *path, ffm_block_structure *bs, struct ffm_parameter param);

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model *model);

ffm_float ffm_cross_validation(struct ffm_problem *prob, ffm_int nr_folds, struct ffm_negative_sampling *ns, struct ffm_block_structure *bs, struct ffm_parameter param);

ffm_block_structure* ffm_read_block_structure(char const *path);

ffm_negative_sampling *ffm_create_negative_sampling(ffm_int negative_position, ffm_int num_negative_samples, char const *path, ffm_int n);

void ffm_destroy_block_structure(ffm_block_structure **bs);

#ifdef __cplusplus
} // namespace mf

} // extern "C"
#endif

#endif // _LIBFFM_H
