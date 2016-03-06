#pragma GCC diagnostic ignored "-Wunused-result" 
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <new>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <cstring>
#include <vector>
#include <pmmintrin.h>

#if defined USEOMP
#include <omp.h>
#endif

#include "ffm.h"
#include "ffm_internal.h"

namespace ffm {

namespace {

using namespace std;

ffm_int const kALIGNByte = 16;
ffm_int const kALIGN = kALIGNByte/sizeof(ffm_float);
ffm_int const kCHUNK_SIZE = 10000000;
ffm_int const kMaxLineSize = 100000;

inline ffm_float wTx(
    ffm_node *begin,
    ffm_node *end,
    ffm_float r,
    ffm_model &model, 
    ffm_float kappa=0, 
    ffm_float eta=0, 
    ffm_float lambda=0, 
    bool do_update=false)
{
    ffm_long align0 = (ffm_long)model.k*2;
    ffm_long align1 = (ffm_long)model.m*align0;

    __m128 XMMkappa = _mm_set1_ps(kappa);
    __m128 XMMeta = _mm_set1_ps(eta);
    __m128 XMMlambda = _mm_set1_ps(lambda);

    __m128 XMMt = _mm_setzero_ps();

    for(ffm_node *N1 = begin; N1 != end; N1++)
    {
        ffm_int j1 = N1->j;
        ffm_int f1 = N1->f;
        ffm_float v1 = N1->v;
        if(j1 >= model.n || f1 >= model.m)
            continue;

        for(ffm_node *N2 = N1+1; N2 != end; N2++)
        {
            ffm_int j2 = N2->j;
            ffm_int f2 = N2->f;
            ffm_float v2 = N2->v;
            if(j2 >= model.n || f2 >= model.m)
                continue;

            ffm_float *w1 = model.W + j1*align1 + f2*align0;
            ffm_float *w2 = model.W + j2*align1 + f1*align0;

            __m128 XMMv = _mm_set1_ps(v1*v2*r);

            if(do_update)
            {
                __m128 XMMkappav = _mm_mul_ps(XMMkappa, XMMv);

                ffm_float *wg1 = w1 + model.k;
                ffm_float *wg2 = w2 + model.k;
                for(ffm_int d = 0; d < model.k; d += 4)
                {
                    __m128 XMMw1 = _mm_load_ps(w1+d);
                    __m128 XMMw2 = _mm_load_ps(w2+d);

                    __m128 XMMwg1 = _mm_load_ps(wg1+d);
                    __m128 XMMwg2 = _mm_load_ps(wg2+d);

                    __m128 XMMg1 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw1),
                                   _mm_mul_ps(XMMkappav, XMMw2));
                    __m128 XMMg2 = _mm_add_ps(
                                   _mm_mul_ps(XMMlambda, XMMw2),
                                   _mm_mul_ps(XMMkappav, XMMw1));

                    XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
                    XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

                    XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
                    XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta, 
                            _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

                    _mm_store_ps(w1+d, XMMw1);
                    _mm_store_ps(w2+d, XMMw2);

                    _mm_store_ps(wg1+d, XMMwg1);
                    _mm_store_ps(wg2+d, XMMwg2);
                }
            }
            else
            {
                for(ffm_int d = 0; d < model.k; d += 4)
                {
                    __m128  XMMw1 = _mm_load_ps(w1+d);
                    __m128  XMMw2 = _mm_load_ps(w2+d);

                    XMMt = _mm_add_ps(XMMt, 
                           _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
                }
            }
        }
    }

    if(do_update)
        return 0;

    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    ffm_float t;
    _mm_store_ss(&t, XMMt);

    return t;
}

// Requires OMP number of threads to have been set before calling this function!
void gradient_descent(bool update_model, ffm_negative_sampling *ns, ffm_block_structure *bs, ffm_int num_neg,
    ffm_float eta, ffm_float lambda, shared_ptr<ffm_model> model,
    ffm_int l, ffm_float *Y, ffm_node *X, ffm_long *P, ffm_float *R, ffm_int *order,
    ffm_double *target_loss, ffm_double *target_accuracy)
{
    vector<ffm_node> example_row;
    ffm_double loss = 0.0;
    ffm_double accuracy = 0.0;
#if defined USEOMP
#pragma omp parallel
#endif
    {
       unsigned long long next_random = 0;

#if defined USEOMP
       next_random = omp_get_thread_num();
#pragma omp for schedule(static) reduction(+: loss) reduction(+: accuracy) private(example_row)
#endif
    for(ffm_int ii = 0; ii < l; ii++)
        for(ffm_int jj = 0; jj < num_neg+1; jj++)
        {
            ffm_int i = ii;
            if(order!=nullptr)
                i = order[ii];

            ffm_float y = Y[i];

            if(jj>0)
                y = -1.0f;

            join_features(&next_random, jj>0, ns, bs, &X[P[i]], P[i+1] - P[i], example_row);

            ffm_float r = R == nullptr ? 1.0 : R[i];

            ffm_float t = wTx(example_row.data(), example_row.data() + example_row.size(), r, *model);

            ffm_float expnyt = exp(-y*t);

            loss += log(1+expnyt);

            if(y*t >= 0.0)
                accuracy += 1.0;

            if(update_model) {
                ffm_float kappa = -y*expnyt/(1+expnyt);
                wTx(example_row.data(), example_row.data() + example_row.size(), r, *model, kappa, eta, lambda, true);
            }
        }
    }

    if(target_loss!=nullptr)
        *target_loss += loss;

    if(target_accuracy!=nullptr)
        *target_accuracy += accuracy;

}

inline ffm_float bs_wTx(
    ffm_block_structure *bs,
    ffm_node *begin,
    ffm_node *end,
    ffm_float r,
    ffm_model &model, 
    ffm_float kappa=0, 
    ffm_float eta=0, 
    ffm_float lambda=0, 
    bool do_update=false)
{
    ffm_long align0 = (ffm_long)model.k*2;
    ffm_long align1 = (ffm_long)model.m*align0;

    __m128 XMMkappa = _mm_set1_ps(kappa);
    __m128 XMMeta = _mm_set1_ps(eta);
    __m128 XMMlambda = _mm_set1_ps(lambda);

    __m128 XMMt = _mm_setzero_ps();

    for(ffm_node *N1 = begin; N1 != end; N1++)
    {
        ffm_int j1 = N1->j;
        ffm_int f1 = N1->f;
        ffm_float v1 = N1->v;
        if(j1 >= model.n || f1 >= model.m)
            continue;

        for(ffm_node *N2 = N1+1; N2 != end; N2++)
        {
            ffm_int j2 = N2->j;
            ffm_int f2 = N2->f;
            ffm_float v2 = N2->v;
            if(j2 >= model.n || f2 >= model.m)
                continue;

            ffm_int base1 = bs->index[j1];
            ffm_int joined1 = 0;
            if(bs!= nullptr && j1 < bs->nr_features)
                joined1 = bs->index[j1+1] - base1;

            for(ffm_int i1=0; i1<(1+joined1); i1++)
            {
                ffm_int jj1 = i1==0? j1 : bs->features[base1+i1-1].j;
                ffm_int ff1 = i1==0? f1 : bs->features[base1+i1-1].f;
                ffm_int vv1 = i1==0? v1 : bs->features[base1+i1-1].v;

                ffm_int base2 = bs->index[j2];
                ffm_int joined2 = 0;
                if(bs!= nullptr && j2 < bs->nr_features)
                    joined2 = bs->index[j2+1] - base2;

                for(ffm_int i2=0; i2<(1+joined2); i2++)
                {
                    ffm_int jj2 = i2==0? j2 : bs->features[base2+i2-1].j;
                    ffm_int ff2 = i2==0? f2 : bs->features[base2+i2-1].f;
                    ffm_int vv2 = i2==0? v2 : bs->features[base2+i2-1].v;

                    ffm_float *w1 = model.W + jj1*align1 + ff2*align0;
                    ffm_float *w2 = model.W + jj2*align1 + ff1*align0;

                    __m128 XMMv = _mm_set1_ps(vv1*vv2*r);

                    if(do_update)
                    {
                        __m128 XMMkappav = _mm_mul_ps(XMMkappa, XMMv);

                        ffm_float *wg1 = w1 + model.k;
                        ffm_float *wg2 = w2 + model.k;
                        for(ffm_int d = 0; d < model.k; d += 4)
                        {
                            __m128 XMMw1 = _mm_load_ps(w1+d);
                            __m128 XMMw2 = _mm_load_ps(w2+d);

                            __m128 XMMwg1 = _mm_load_ps(wg1+d);
                            __m128 XMMwg2 = _mm_load_ps(wg2+d);

                            __m128 XMMg1 = _mm_add_ps(
                                           _mm_mul_ps(XMMlambda, XMMw1),
                                           _mm_mul_ps(XMMkappav, XMMw2));
                            __m128 XMMg2 = _mm_add_ps(
                                           _mm_mul_ps(XMMlambda, XMMw2),
                                           _mm_mul_ps(XMMkappav, XMMw1));

                            XMMwg1 = _mm_add_ps(XMMwg1, _mm_mul_ps(XMMg1, XMMg1));
                            XMMwg2 = _mm_add_ps(XMMwg2, _mm_mul_ps(XMMg2, XMMg2));

                            XMMw1 = _mm_sub_ps(XMMw1, _mm_mul_ps(XMMeta, 
                                    _mm_mul_ps(_mm_rsqrt_ps(XMMwg1), XMMg1)));
                            XMMw2 = _mm_sub_ps(XMMw2, _mm_mul_ps(XMMeta, 
                                    _mm_mul_ps(_mm_rsqrt_ps(XMMwg2), XMMg2)));

                            _mm_store_ps(w1+d, XMMw1);
                            _mm_store_ps(w2+d, XMMw2);

                            _mm_store_ps(wg1+d, XMMwg1);
                            _mm_store_ps(wg2+d, XMMwg2);
                        }
                    }
                    else
                    {
                        for(ffm_int d = 0; d < model.k; d += 4)
                        {
                            __m128  XMMw1 = _mm_load_ps(w1+d);
                            __m128  XMMw2 = _mm_load_ps(w2+d);

                            XMMt = _mm_add_ps(XMMt, 
                                   _mm_mul_ps(_mm_mul_ps(XMMw1, XMMw2), XMMv));
                        }
                    }

                }


            }

        }
    }

    if(do_update)
        return 0;

    XMMt = _mm_hadd_ps(XMMt, XMMt);
    XMMt = _mm_hadd_ps(XMMt, XMMt);
    ffm_float t;
    _mm_store_ss(&t, XMMt);

    return t;
}

// Requires OMP number of threads to have been set before calling this function!
void bs_gradient_descent(bool update_model, ffm_negative_sampling *ns, ffm_block_structure *bs, ffm_int num_neg,
    ffm_float eta, ffm_float lambda, shared_ptr<ffm_model> model,
    ffm_int l, ffm_float *Y, ffm_node *X, ffm_long *P, ffm_float *R, ffm_int *order,
    ffm_double *target_loss, ffm_double *target_accuracy)
{
    vector<ffm_node> example_row;
    ffm_double loss = 0.0;
    ffm_double accuracy = 0.0;
#if defined USEOMP
#pragma omp parallel
#endif
    {
       unsigned long long next_random = 0;

#if defined USEOMP
       next_random = omp_get_thread_num();
#pragma omp for schedule(static) reduction(+: loss) reduction(+: accuracy) private(example_row)
#endif
    for(ffm_int ii = 0; ii < l; ii++)
    {
        ffm_int i = ii;
        if(order!=nullptr)
            i = order[ii];

        example_row.resize(P[i+1] - P[i]);
        copy(&X[P[i]], &X[P[i+1]], example_row.begin());

        for(ffm_int jj = 0; jj < num_neg+1; jj++)
        {

            ffm_float y = Y[i];

            if(jj>0) // negative sampling
            {
                next_random = next_random * (unsigned long long)25214903917 + 11;
                example_row[ns->negative_position].j = ns->sampling_buckets[next_random % ns->num_sampling_buckets];
                y = -1.0f;
            }

            ffm_float r = R == nullptr ? 1.0 : R[i];

            ffm_float t = bs_wTx(bs, example_row.data(), example_row.data() + example_row.size(), r, *model);

            ffm_float expnyt = exp(-y*t);

            loss += log(1+expnyt);

            if(y*t >= 0.0)
                accuracy += 1.0;

            if(update_model) {
                ffm_float kappa = -y*expnyt/(1+expnyt);
                bs_wTx(bs, example_row.data(), example_row.data() + example_row.size(), r, *model, kappa, eta, lambda, true);
            }
        }
    }
    }

    if(target_loss!=nullptr)
        *target_loss += loss;

    if(target_accuracy!=nullptr)
        *target_accuracy += accuracy;

}

ffm_float* malloc_aligned_float(ffm_long size)
{
    void *ptr;

#ifdef _WIN32
    ptr = _aligned_malloc(size*sizeof(ffm_float), kALIGNByte);
    if(ptr == nullptr)
        throw bad_alloc();
#else
    int status = posix_memalign(&ptr, kALIGNByte, size*sizeof(ffm_float));
    if(status != 0)
        throw bad_alloc();
#endif
    
    return (ffm_float*)ptr;
}

ffm_model* init_model(ffm_int n, ffm_int m, ffm_parameter param)
{
    ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k/kALIGN)*kALIGN;

    ffm_model *model = new ffm_model;
    model->n = n;
    model->k = k_aligned;
    model->m = m;
    model->W = nullptr;
    model->normalization = param.normalization;
    
    try
    {
        model->W = malloc_aligned_float((ffm_long)n*m*k_aligned*2);
    }
    catch(bad_alloc const &e)
    {
        ffm_destroy_model(&model);
        throw;
    }

    ffm_float coef = 0.5/sqrt(param.k);
    ffm_float *w = model->W;

    default_random_engine generator;
    uniform_real_distribution<ffm_float> distribution(0.0, 1.0);

    for(ffm_int j = 0; j < model->n; j++)
    {
        for(ffm_int f = 0; f < model->m; f++)
        {
            for(ffm_int d = 0; d < param.k; d++, w++)
                *w = coef*distribution(generator);
            for(ffm_int d = param.k; d < k_aligned; d++, w++)
                *w = 0;
            for(ffm_int d = k_aligned; d < 2*k_aligned; d++, w++)
                *w = 1;
        }
    }

    return model;
}

void shrink_model(ffm_model &model, ffm_int k_new)
{
    for(ffm_int j = 0; j < model.n; j++)
    {
        for(ffm_int f = 0; f < model.m; f++)
        {
            ffm_float *src = model.W + ((ffm_long)j*model.m+f)*model.k*2;
            ffm_float *dst = model.W + ((ffm_long)j*model.m+f)*k_new;
            copy(src, src+k_new, dst);
        }
    }

    model.k = k_new;
}

inline ffm_float example_scale(ffm_block_structure *bs, ffm_node *p, ffm_int n, vector<ffm_node> &example_row)
{
    ffm_float norm = 0;
    join_features(bs, p, n, example_row);
    for(auto p = example_row.begin(); p < example_row.end(); p++)
        norm += p->v*p->v;
    return norm;
}

vector<ffm_float> normalize(ffm_block_structure *bs, ffm_problem &prob)
{
    vector<ffm_float> R(prob.l);
    vector<ffm_node> example_row;
#if defined USEOMP
#pragma omp parallel for schedule(static) private(example_row)
#endif
    for(ffm_int i = 0; i < prob.l; i++)
        R[i] = 1 / example_scale(bs, &prob.X[prob.P[i]], prob.P[i+1] - prob.P[i], example_row);

    return R;
}

shared_ptr<ffm_model> train(
    ffm_problem *tr, 
    vector<ffm_int> &order, 
    ffm_negative_sampling *ns,
    ffm_block_structure *bs,
    ffm_parameter param, 
    ffm_problem *va=nullptr)
{
#if defined USEOMP
    ffm_int old_nr_threads = omp_get_num_threads();
    omp_set_num_threads(param.nr_threads);
#endif

    ffm_int n=tr->n, m=tr->m;
    if(bs!=nullptr) {
      n = max(n, bs->max_feature);
      m = max(m, bs->max_field);
    }

    ffm_int num_neg = 0;
    if(ns!=nullptr)
        num_neg = ns->num_negative_samples;

    shared_ptr<ffm_model> model = 
        shared_ptr<ffm_model>(init_model(n, m, param),
            [] (ffm_model *ptr) { ffm_destroy_model(&ptr); });

    vector<ffm_float> R_tr, R_va;
    if(param.normalization)
    {
        R_tr = normalize(bs, *tr);
        if(va != nullptr)
            R_va = normalize(bs, *va);
    }
    else
    {
        R_tr = vector<ffm_float>(tr->l, 1);
        if(va != nullptr)
            R_va = vector<ffm_float>(va->l, 1);
    }

    bool auto_stop = param.auto_stop && va != nullptr && va->l != 0;

    ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k/kALIGN)*kALIGN;
    ffm_long w_size = (ffm_long)model->n * model->m * k_aligned * 2;
    vector<ffm_float> prev_W;
    if(auto_stop)
        prev_W.assign(w_size, 0);
    ffm_double best_va_loss = numeric_limits<ffm_double>::max();

    if(!param.quiet)
    {
        if(param.auto_stop && (va == nullptr || va->l == 0))
            cerr << "warning: ignoring auto-stop because there is no validation set" << endl;

        cout.width(4);
        cout << "iter";
        cout.width(13);
        cout << "tr_logloss";
        if(va != nullptr && va->l != 0)
        {
            cout.width(13);
            cout << "va_logloss";
            cout.width(13);
            cout << "va_accuracy";
        }
        cout << endl;
    }

    for(ffm_int iter = 1; iter <= param.nr_iters; iter++)
    {
        ffm_double tr_loss = 0;
        vector<ffm_node> example_row;
        if(param.random)
            random_shuffle(order.begin(), order.end());

        bs_gradient_descent(true, ns, bs, num_neg,
            param.eta, param.lambda, model,
            tr->l, tr->Y, tr->X, tr->P, R_tr.data(), order.data(),
            &tr_loss, nullptr);

        tr_loss /= tr->l * (1 + num_neg);

        if(!param.quiet)
        {
            cout.width(4);
            cout << iter;
            cout.width(13);
            cout << fixed << setprecision(5) << tr_loss;
            if(va != nullptr && va->l != 0)
            {
                ffm_double va_loss = 0;
                ffm_double va_accuracy = 0;

                bs_gradient_descent(false, ns, bs, num_neg,
                    param.eta, param.lambda, model,
                    va->l, va->Y, va->X, va->P, R_va.data(), nullptr,
                    &va_loss, &va_accuracy);

                va_loss /= va->l * (1 + num_neg);
                va_accuracy /= va->l * (1 + num_neg);

                cout.width(13);
                cout << fixed << setprecision(5) << va_loss;
                cout.width(13);
                cout << fixed << setprecision(5) << va_accuracy;

                if(auto_stop)
                {
                    if(va_loss > best_va_loss)
                    {
                        memcpy(model->W, prev_W.data(), w_size*sizeof(ffm_float));
                        cout << endl << "Auto-stop. Use model at " << iter-1 << "th iteration." << endl;
                        break;
                    }
                    else
                    {
                        memcpy(prev_W.data(), model->W, w_size*sizeof(ffm_float));
                        best_va_loss = va_loss; 
                    }
                }
            }
            cout << endl;
        }
    }

    shrink_model(*model, param.k);

#if defined USEOMP
    omp_set_num_threads(old_nr_threads);
#endif

    return model;
}

// TODO: This function will be merged with train().
shared_ptr<ffm_model> train_on_disk(
    string tr_path,
    string va_path,
    ffm_negative_sampling *ns,
    ffm_block_structure *bs,
    ffm_parameter param)
{
#if defined USEOMP
    ffm_int old_nr_threads = omp_get_num_threads();
    omp_set_num_threads(param.nr_threads);
#endif

    FILE *f_tr = fopen(tr_path.c_str(), "rb");
    FILE *f_va = nullptr;
    if(!va_path.empty())
        f_va = fopen(va_path.c_str(), "rb");

    ffm_int m, n, max_l;
    ffm_long max_nnz;
    fread(&m, sizeof(ffm_int), 1, f_tr);
    fread(&n, sizeof(ffm_int), 1, f_tr);
    fread(&max_l, sizeof(ffm_int), 1, f_tr);
    fread(&max_nnz, sizeof(ffm_long), 1, f_tr);

    if(bs!=nullptr) {
      n = max(n, bs->max_feature);
      m = max(m, bs->max_field);
    }

    ffm_int num_neg = 0;
    if(ns!=nullptr)
        num_neg = ns->num_negative_samples;

    shared_ptr<ffm_model> model = 
        shared_ptr<ffm_model>(init_model(n, m, param),
            [] (ffm_model *ptr) { ffm_destroy_model(&ptr); });

    vector<ffm_float> Y;
    Y.reserve(max_l);
    vector<ffm_float> R;
    R.reserve(max_l);
    vector<ffm_long> P;
    P.reserve(max_l+1);
    vector<ffm_node> X;
    X.reserve(max_nnz);

    bool auto_stop = param.auto_stop && !va_path.empty();

    ffm_int k_aligned = (ffm_int)ceil((ffm_double)param.k/kALIGN)*kALIGN;
    ffm_long w_size = (ffm_long)model->n * model->m * k_aligned * 2;
    vector<ffm_float> prev_W;
    if(auto_stop)
        prev_W.assign(w_size, 0);
    ffm_double best_va_loss = numeric_limits<ffm_double>::max();

    if(!param.quiet)
    {
        if(param.auto_stop && va_path.empty())
            cerr << "warning: ignoring auto-stop because there is no validation set" << endl;
        cout.width(4);
        cout << "iter";
        cout.width(13);
        cout << "tr_logloss";
        if(!va_path.empty())
        {
            cout.width(13);
            cout << "va_logloss";
            cout.width(13);
            cout << "va_accuracy";
        }
        cout << endl;
    }

    for(ffm_int iter = 1; iter <= param.nr_iters; iter++)
    {
        ffm_double tr_loss = 0;
        vector<ffm_node> example_row;

        fseek(f_tr, 3*sizeof(ffm_int)+sizeof(ffm_long), SEEK_SET);

        ffm_int tr_l = 0;
        while(true)
        {
            ffm_int l;
            fread(&l, sizeof(ffm_int), 1, f_tr);
            tr_l += l;
            if(l == 0)
                break;

            Y.resize(l);
            fread(Y.data(), sizeof(ffm_float), l, f_tr);

            R.resize(l);
            fread(R.data(), sizeof(ffm_float), l, f_tr);

            P.resize(l+1);
            fread(P.data(), sizeof(ffm_long), l+1, f_tr);

            X.resize(P[l]);
            fread(X.data(), sizeof(ffm_node), P[l], f_tr);

            ffm_float *rdata = param.normalization ? R.data() : nullptr;

            bs_gradient_descent(true, ns, bs, num_neg,
                param.eta, param.lambda, model,
                l, Y.data(), X.data(), P.data(), rdata, nullptr,
                &tr_loss, nullptr);

        }

        if(!param.quiet)
        {
            tr_loss /= tr_l * (1 + num_neg);

            cout.width(4);
            cout << iter;
            cout.width(13);
            cout << fixed << setprecision(5) << tr_loss;

            if(f_va != nullptr)
            {
                fseek(f_va, 3*sizeof(ffm_int)+sizeof(ffm_long), SEEK_SET);

                ffm_int va_l = 0;
                ffm_double va_loss = 0;
                ffm_double va_accuracy = 0;
                while(true)
                {
                    ffm_int l;
                    fread(&l, sizeof(ffm_int), 1, f_va);
                    va_l += l;
                    if(l == 0)
                        break;

                    vector<ffm_float> Y(l);
                    fread(Y.data(), sizeof(ffm_float), l, f_va);

                    vector<ffm_float> R(l);
                    fread(R.data(), sizeof(ffm_float), l, f_va);

                    vector<ffm_long> P(l+1);
                    fread(P.data(), sizeof(ffm_long), l+1, f_va);

                    vector<ffm_node> X(P[l]);
                    fread(X.data(), sizeof(ffm_node), P[l], f_va);

                    ffm_float *rdata = param.normalization ? R.data() : nullptr;

                    bs_gradient_descent(false, ns, bs, num_neg,
                        param.eta, param.lambda, model,
                        l, Y.data(), X.data(), P.data(), rdata, nullptr,
                        &va_loss, &va_accuracy);

                }

                va_loss /= va_l * (1 + num_neg);
                va_accuracy /= va_l * (1 + num_neg);

                cout.width(13);
                cout << fixed << setprecision(5) << va_loss;
                cout.width(13);
                cout << fixed << setprecision(5) << va_accuracy;

                if(auto_stop)
                {
                    if(va_loss > best_va_loss)
                    {
                        memcpy(model->W, prev_W.data(), w_size*sizeof(ffm_float));
                        cout << endl << "Auto-stop. Use model at " << iter-1 << "th iteration." << endl;
                        break;
                    }
                    else
                    {
                        memcpy(prev_W.data(), model->W, w_size*sizeof(ffm_float));
                        best_va_loss = va_loss; 
                    }
                }
            }
            cout << endl;
        }
    }

    shrink_model(*model, param.k);

    fclose(f_tr);
    if(!va_path.empty())
        fclose(f_va);

#if defined USEOMP
    omp_set_num_threads(old_nr_threads);
#endif

    return model;
}

} // unnamed namespace

ffm_problem* ffm_read_problem(char const *path)
{
    if(strlen(path) == 0)
        return nullptr;

    FILE *f = fopen(path, "r");
    if(f == nullptr)
        return nullptr;

    ffm_problem *prob = new ffm_problem;
    prob->l = 0;
    prob->n = 0;
    prob->m = 0;
    prob->X = nullptr;
    prob->P = nullptr;
    prob->Y = nullptr;

    char line[kMaxLineSize];

    ffm_long nnz = 0;
    for(; fgets(line, kMaxLineSize, f) != nullptr; prob->l++)
    {
        strtok(line, " \t");
        for(; ; nnz++)
        {
            char *ptr = strtok(nullptr," \t");
            if(ptr == nullptr || *ptr == '\n')
                break;
        }
    }
    rewind(f);

    prob->X = new ffm_node[nnz];
    prob->P = new ffm_long[prob->l+1];
    prob->Y = new ffm_float[prob->l];

    ffm_long p = 0;
    prob->P[0] = 0;
    for(ffm_int i = 0; fgets(line, kMaxLineSize, f) != nullptr; i++)
    {
        char *y_char = strtok(line, " \t");
        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;
        prob->Y[i] = y;

        for(; ; p++)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_int field = atoi(field_char);
            ffm_int idx = atoi(idx_char);
            ffm_float value = atof(value_char);

            prob->m = max(prob->m, field+1);
            prob->n = max(prob->n, idx+1);
            
            prob->X[p].f = field;
            prob->X[p].j = idx;
            prob->X[p].v = value;
        }
        prob->P[i+1] = p;
    }

    fclose(f);

    return prob;
}

int ffm_read_problem_to_disk(ffm_block_structure *bs, char const *txt_path, char const *bin_path)
{
    FILE *f_txt = fopen(txt_path, "r");
    if(f_txt == nullptr)
        return 1;

    FILE *f_bin = fopen(bin_path, "wb");
    if(f_bin == nullptr)
        return 1;

    vector<char> line(kMaxLineSize);

    ffm_int m = 0;
    ffm_int n = 0;
    ffm_int max_l = 0;
    ffm_long max_nnz = 0;
    ffm_long p = 0;

    vector<ffm_float> Y;
    vector<ffm_float> R;
    vector<ffm_long> P(1, 0);
    vector<ffm_node> X;

    auto write_chunk = [&] ()
    {
        ffm_int l = Y.size();
        ffm_long nnz = P[l];

        max_l = max(max_l, l);
        max_nnz = max(max_nnz, nnz);

        fwrite(&l, sizeof(ffm_int), 1, f_bin);
        fwrite(Y.data(), sizeof(ffm_float), l, f_bin);
        fwrite(R.data(), sizeof(ffm_float), l, f_bin);
        fwrite(P.data(), sizeof(ffm_long), l+1, f_bin);
        fwrite(X.data(), sizeof(ffm_node), nnz, f_bin);

        Y.clear();
        R.clear();
        P.assign(1, 0);
        X.clear();
        p = 0;
    };

    fwrite(&m, sizeof(ffm_int), 1, f_bin);
    fwrite(&n, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_l, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_nnz, sizeof(ffm_long), 1, f_bin);

    vector<ffm_node> example_row;

    while(fgets(line.data(), kMaxLineSize, f_txt))
    {
        char *y_char = strtok(line.data(), " \t");

        ffm_float y = (atoi(y_char)>0)? 1.0f : -1.0f;

        for(; ; p++)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_node N;
            N.f = atoi(field_char);
            N.j = atoi(idx_char);
            N.v = atof(value_char);

            X.push_back(N);

            m = max(m, N.f+1);
            n = max(n, N.j+1);
        }

        ffm_int example_size = p - P[P.size()-1];
        ffm_float scale = 1 / example_scale(bs, &X[X.size()-example_size], example_size, example_row);

        Y.push_back(y);
        R.push_back(scale);
        P.push_back(p);

        if(X.size() > (size_t)kCHUNK_SIZE)
            write_chunk(); 
    }
    write_chunk(); 
    write_chunk(); 

    rewind(f_bin);
    fwrite(&m, sizeof(ffm_int), 1, f_bin);
    fwrite(&n, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_l, sizeof(ffm_int), 1, f_bin);
    fwrite(&max_nnz, sizeof(ffm_long), 1, f_bin);

    fclose(f_bin);
    fclose(f_txt);

    return 0;
}

void ffm_destroy_problem(ffm_problem **prob)
{
    if(prob == nullptr || *prob == nullptr)
        return;
    delete[] (*prob)->X;
    delete[] (*prob)->P;
    delete[] (*prob)->Y;
    delete *prob;
    *prob = nullptr;
}

ffm_int ffm_save_model(ffm_model *model, char const *path)
{
    ofstream f_out(path);
    if(!f_out.is_open())
        return 1;

    f_out << "n " << model->n << "\n";
    f_out << "m " << model->m << "\n";
    f_out << "k " << model->k << "\n";
    f_out << "normalization " << model->normalization << "\n";

    ffm_float *ptr = model->W;
    for(ffm_int j = 0; j < model->n; j++)
    {
        for(ffm_int f = 0; f < model->m; f++)
        {
            f_out << "w" << j << "," << f << " ";
            for(ffm_int d = 0; d < model->k; d++, ptr++)
                f_out << *ptr << " ";
            f_out << "\n";
        }
    }

    return 0;
}

ffm_model* ffm_load_model(char const *path)
{
    ifstream f_in(path);
    if(!f_in.is_open())
        return nullptr;

    string dummy;

    ffm_model *model = new ffm_model;
    model->W = nullptr;

    f_in >> dummy >> model->n >> dummy >> model->m >> dummy >> model->k 
         >> dummy >> model->normalization;

    try
    {
        model->W = malloc_aligned_float((ffm_long)model->m*model->n*model->k);
    }
    catch(bad_alloc const &e)
    {
        ffm_destroy_model(&model);
        return nullptr;
    }

    ffm_float *ptr = model->W;
    for(ffm_int j = 0; j < model->n; j++)
    {
        for(ffm_int f = 0; f < model->m; f++)
        {
            f_in >> dummy;
            for(ffm_int d = 0; d < model->k; d++, ptr++)
                f_in >> *ptr;
        }
    }

    return model;
}

void ffm_destroy_model(ffm_model **model)
{
    if(model == nullptr || *model == nullptr)
        return;
#ifdef _WIN32
    _aligned_free((*model)->W);
#else
    free((*model)->W);
#endif
    delete *model;
    *model = nullptr;
}

void ffm_destroy_block_structure(ffm_block_structure **bs)
{
    if(bs == nullptr || *bs == nullptr)
        return;
    delete (*bs)->index;
    delete (*bs)->features;
    delete *bs;
    *bs = nullptr;
}

ffm_parameter ffm_get_default_param()
{
    ffm_parameter param;

    param.eta = 0.2;
    param.lambda = 0.00002;
    param.nr_iters = 15;
    param.k = 4;
    param.nr_threads = 1;
    param.quiet = false;
    param.normalization = true;
    param.random = true;
    param.auto_stop = false;
    param.negative_samples = 1;
    param.negative_position = -1;

    return param;
}

ffm_model* ffm_train_with_validation(ffm_problem *tr, ffm_problem *va, ffm_negative_sampling *ns, ffm_block_structure *bs, ffm_parameter param)
{
    vector<ffm_int> order(tr->l);
    for(ffm_int i = 0; i < tr->l; i++)
        order[i] = i;

    shared_ptr<ffm_model> model = train(tr, order, ns, bs, param, va);

    ffm_model *model_ret = new ffm_model;

    model_ret->n = model->n;
    model_ret->m = model->m;
    model_ret->k = model->k;
    model_ret->normalization = model->normalization;

    model_ret->W = model->W;
    model->W = nullptr;

    return model_ret;
}

ffm_model* ffm_train(ffm_problem *prob, ffm_parameter param, ffm_negative_sampling *ns, ffm_block_structure *bs)
{
    return ffm_train_with_validation(prob, nullptr, ns, bs, param);
}

ffm_model* ffm_train_with_validation_on_disk(
    char const *tr_path,
    char const *va_path,
    ffm_negative_sampling *ns,
    ffm_block_structure *bs,
    ffm_parameter param)
{
    shared_ptr<ffm_model> model = train_on_disk(tr_path, va_path, ns, bs, param);

    ffm_model *model_ret = new ffm_model;

    model_ret->n = model->n;
    model_ret->m = model->m;
    model_ret->k = model->k;
    model_ret->normalization = model->normalization;

    model_ret->W = model->W;
    model->W = nullptr;

    return model_ret;
}

ffm_model* ffm_train_on_disk(char const *prob_path, ffm_negative_sampling *ns, ffm_block_structure *bs, ffm_parameter param)
{
    return ffm_train_with_validation_on_disk(prob_path, "", ns, bs, param);
}

ffm_float ffm_predict(ffm_node *begin, ffm_node *end, ffm_model *model)
{
    ffm_float r = 1;
    if(model->normalization)
    {
        r = 0;
        for(ffm_node *N = begin; N != end; N++)
            r += N->v*N->v; 
        r = 1/r;
    }

    ffm_long align0 = (ffm_long)model->k;
    ffm_long align1 = (ffm_long)model->m*align0;

    ffm_float t = 0;
    for(ffm_node *N1 = begin; N1 != end; N1++)
    {
        ffm_int j1 = N1->j;
        ffm_int f1 = N1->f;
        ffm_float v1 = N1->v;
        if(j1 >= model->n || f1 >= model->m)
            continue;

        for(ffm_node *N2 = N1+1; N2 != end; N2++)
        {
            ffm_int j2 = N2->j;
            ffm_int f2 = N2->f;
            ffm_float v2 = N2->v;
            if(j2 >= model->n || f2 >= model->m)
                continue;

            ffm_float *w1 = model->W + j1*align1 + f2*align0;
            ffm_float *w2 = model->W + j2*align1 + f1*align0;

            ffm_float v = v1*v2*r;

            for(ffm_int d = 0; d < model->k; d++)
                t += w1[d]*w2[d]*v;
        }
    }

    return 1/(1+exp(-t));
}

ffm_float ffm_cross_validation(
    ffm_problem *prob, 
    ffm_int nr_folds,
    ffm_negative_sampling *ns,
    ffm_block_structure *bs,
    ffm_parameter param)
{
#if defined USEOMP
    ffm_int old_nr_threads = omp_get_num_threads();
    omp_set_num_threads(param.nr_threads);
#endif

    bool quiet = param.quiet;
    param.quiet = true;

    vector<ffm_int> order(prob->l);
    for(ffm_int i = 0; i < prob->l; i++)
        order[i] = i;
    random_shuffle(order.begin(), order.end());

    if(!quiet)
    {
        cout.width(4);
        cout << "fold";
        cout.width(13);
        cout << "logloss";
        cout << endl;
    }

    ffm_double loss = 0;
    ffm_int nr_instance_per_fold = prob->l/nr_folds;
    for(ffm_int fold = 0; fold < nr_folds; fold++)
    {
        ffm_int begin = fold*nr_instance_per_fold;
        ffm_int end = min(begin + nr_instance_per_fold, prob->l);

        vector<ffm_int> order1;
        for(ffm_int i = 0; i < begin; i++)
            order1.push_back(order[i]);
        for(ffm_int i = end; i < prob->l; i++)
            order1.push_back(order[i]);

        shared_ptr<ffm_model> model = train(prob, order1, ns, bs, param);

        ffm_double loss1 = 0;
#if defined USEOMP
#pragma omp parallel for schedule(static) reduction(+: loss1)
#endif
        for(ffm_int ii = begin; ii < end; ii++)
        {
            ffm_int i = order[ii];

            ffm_float y = prob->Y[i];
            
            ffm_node *begin = &prob->X[prob->P[i]];

            ffm_node *end = &prob->X[prob->P[i+1]];

            ffm_float y_bar = ffm_predict(begin, end, model.get());

            loss1 -= y==1? log(y_bar) : log(1-y_bar);
        }
        loss += loss1;

        if(!quiet)
        {
            cout.width(4);
            cout << fold;
            cout.width(13);
            cout << fixed << setprecision(4) << loss1 / (end-begin);
            cout << endl;
        }
    }

    if(!quiet)
    {
        cout.width(17);
        cout.fill('=');
        cout << "" << endl;
        cout.fill(' ');
        cout.width(4);
        cout << "avg";
        cout.width(13);
        cout << fixed << setprecision(4) << loss/prob->l;
        cout << endl;
    }

#if defined USEOMP
    omp_set_num_threads(old_nr_threads);
#endif

    return loss/prob->l;
}

ffm_negative_sampling *ffm_create_negative_sampling(ffm_int negative_position, ffm_int num_negative_samples, char const *path, ffm_int n)
{
    ffm_negative_sampling *res = new ffm_negative_sampling;
    if(res==nullptr)
        throw bad_alloc();

    res->num_sampling_buckets = n;
    res->sampling_buckets = read_negative_probabilities(path, n);
    res->negative_position = negative_position;
    res->num_negative_samples = num_negative_samples;

    return res;
}

ffm_int *read_negative_probabilities(char const *path, ffm_int n)
{
    FILE *f_probs = fopen(path, "rb");
    if(f_probs == nullptr)
        return nullptr;

    char line[kMaxLineSize];
    ffm_float total = 0.0;
    vector< pair<ffm_int, ffm_float> > probabilities;

    for(ffm_int i = 0; fgets(line, kMaxLineSize, f_probs) != nullptr; i++)
    {
        char *key_f_char = strtok(line, " \t");
        char *value_char = strtok(nullptr,"\n");

        if(key_f_char == nullptr || *key_f_char == '\n')
            break;

        ffm_int key_feature = atoi(key_f_char);
        ffm_float value = atof(value_char);
        probabilities.push_back( pair<ffm_int, ffm_float>(key_feature, value) );
        total += value;
    }

    fclose(f_probs);

    sort(probabilities.begin(), probabilities.end(),
         [](const pair<ffm_int, ffm_float> & a, const pair<ffm_int, ffm_float> & b) -> bool
         {
             return a.second < b.second;
         });

    ffm_int *res = new ffm_int[n];
    if(res==nullptr)
        throw bad_alloc();

    auto it = probabilities.begin();
    ffm_float p = it->second / ffm_float(total);
    for(int i=0; i<n; i++) {
      res[i] = it->first;
      if(i >= p * n) {
        it++;
        if(it==probabilities.end()) it = it-1;
        p += it->second / ffm_float(total);
      }
    }

    return res;
}

ffm_block_structure* ffm_read_block_structure(char const *path)
{
    FILE *f_block = fopen(path, "rb");
    if(f_block == nullptr)
        return nullptr;

    char line[kMaxLineSize];

    vector<ffm_int> block_index;
    vector<ffm_node> feature_block;
    ffm_int last_key_feature = -1;
    ffm_int max_feature = -1;
    ffm_int max_field = -1;

    for(ffm_int i = 0; fgets(line, kMaxLineSize, f_block) != nullptr; i++)
    {
        char *key_f_char = strtok(line, " \t");
        ffm_int key_feature = atoi(key_f_char);

        if(key_feature <= last_key_feature) {
            cerr << "error: block structure key features are not in strictly ascending order.";
            fclose(f_block);
            return nullptr;
        }

        last_key_feature = key_feature;
        block_index.resize(key_feature+1, feature_block.size());

        for(; ;)
        {
            char *field_char = strtok(nullptr,":");
            char *idx_char = strtok(nullptr,":");
            char *value_char = strtok(nullptr," \t");
            if(field_char == nullptr || *field_char == '\n')
                break;

            ffm_int field = atoi(field_char);
            ffm_int idx = atoi(idx_char);
            ffm_float value = atof(value_char);
            max_feature = max(max_feature, idx+1);
            max_field = max(max_field, field+1);

            feature_block.push_back({field, idx, value});
        }
    }

    block_index.resize(last_key_feature+2, feature_block.size());
    max_feature = max(max_feature, last_key_feature+1);

    fclose(f_block);

    ffm_block_structure * block = new ffm_block_structure;
    if(block==nullptr)
        throw bad_alloc();

    block->nr_features = last_key_feature+1;
    block->max_feature = max_feature;
    block->max_field = max_field;

    block->index = new ffm_int[block_index.size()];
    if(block->index==nullptr)
        throw bad_alloc();
    std::copy(block_index.begin(), block_index.end(), block->index);

    block->features = new ffm_node[feature_block.size()];
    if(block->features==nullptr)
        throw bad_alloc();
    std::copy(feature_block.begin(), feature_block.end(), block->features);

    return block;
}

} // namespace ffm
