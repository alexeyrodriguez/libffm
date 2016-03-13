#ifndef _LIBFFM_INTERNAL_H
#define _LIBFFM_INTERNAL_H

#include <pmmintrin.h>
#include <map>
#include <random>
#include "ffm.h"

using namespace std;

namespace ffm
{

inline void join_features(unsigned long long *next_random,
                          const bool negative_sample, ffm_negative_sampling *ns, ffm_block_structure *bs,
                          ffm_node *begin, ffm_int size, vector<ffm_node> &target_features)
{
  target_features.clear();
  for(ffm_int i=0; i<size; i++)
  {
    ffm_node *p = begin + i;
    target_features.push_back(*p);

    if(negative_sample && i==ns->negative_position)
    {
      *next_random = *next_random * (unsigned long long)25214903917 + 11;
      target_features.back().j = ns->sampling_buckets[*next_random % ns->num_sampling_buckets];
    }

    ffm_int feature = target_features.back().j;

    if(bs != nullptr && feature < bs->nr_features) {
      for(int j = bs->index[feature]; j < bs->index[feature+1]; j++)
        target_features.push_back(bs->features[j]);
    }
  }
}

inline ffm_float sq_cosine(ffm_block_structure *bs, ffm_model model, ffm_node n1, ffm_node n2)
{
    ffm_long align0 = (ffm_long)model.k*2;
    ffm_long align1 = (ffm_long)model.m*align0;

    __m128 XMMs = _mm_setzero_ps();
    __m128 XMMl1 = _mm_setzero_ps();
    __m128 XMMl2 = _mm_setzero_ps();

    ffm_int base1 = 0;
    ffm_int joined1 = 0;
    if(bs!= nullptr && n1.j < bs->nr_features) {
        base1 = bs->index[n1.j];
        joined1 = bs->index[n1.j+1] - base1;
    }

    ffm_int base2 = 0;
    ffm_int joined2 = 0;
    if(bs!= nullptr && n2.j < bs->nr_features) {
        base2 = bs->index[n2.j];
        joined2 = bs->index[n2.j+1] - base2;
    }

    for(ffm_int d = 0; d < model.k; d += 4)
    {
        __m128  XMMe1 = _mm_setzero_ps();
        __m128  XMMe2 = _mm_setzero_ps();

        for(ffm_int i1=0; i1<(1+joined1); i1++)
        {
            ffm_int jj1 = i1==0? n1.j : bs->features[base1+i1-1].j;
            ffm_float vv1 = i1==0? n1.v : bs->features[base1+i1-1].v;

            __m128 XMMv1 = _mm_set1_ps(vv1);

            ffm_float *w1 = model.W + jj1*align1 + 0; //ff2*align0;

            __m128  XMMw1 = _mm_mul_ps(_mm_load_ps(w1+d), XMMv1);
            XMMe1 = _mm_add_ps(XMMe1, XMMw1);
        }

        for(ffm_int i2=0; i2<(1+joined2); i2++)
        {
            ffm_int jj2 = i2==0? n2.j : bs->features[base2+i2-1].j;
            ffm_float vv2 = i2==0? n2.v : bs->features[base2+i2-1].v;

            __m128 XMMv2 = _mm_set1_ps(vv2);

            ffm_float *w2 = model.W + jj2*align1 + 0; //ff1*align0;

            __m128  XMMw2 = _mm_mul_ps(_mm_load_ps(w2+d), XMMv2);
            XMMe2 = _mm_add_ps(XMMe2, XMMw2);
        }

        XMMs = _mm_add_ps(XMMs, _mm_mul_ps(XMMe1, XMMe2));
        XMMl1 = _mm_add_ps(XMMl1, _mm_mul_ps(XMMe1, XMMe1));
        XMMl2 = _mm_add_ps(XMMl2, _mm_mul_ps(XMMe2, XMMe2));

    }

    XMMs = _mm_hadd_ps(XMMs, XMMs);
    XMMs = _mm_hadd_ps(XMMs, XMMs);
    ffm_float s;
    _mm_store_ss(&s, XMMs);

    XMMl1 = _mm_hadd_ps(XMMl1, XMMl1);
    XMMl1 = _mm_hadd_ps(XMMl1, XMMl1);
    ffm_float l1;
    _mm_store_ss(&l1, XMMl1);

    XMMl2 = _mm_hadd_ps(XMMl2, XMMl2);
    XMMl2 = _mm_hadd_ps(XMMl2, XMMl2);
    ffm_float l2;
    _mm_store_ss(&l2, XMMl2);

    return s*s/l1/l2;
}


inline void join_features(ffm_block_structure *bs, ffm_node *begin, ffm_int size, vector<ffm_node> &target_features)
{
    join_features(nullptr, false, nullptr, bs, begin, size, target_features);
}

void read_negative_probabilities(char const *path, ffm_int n, ffm_int **buckets, ffm_int *b, ffm_int **uni_buckets);

}

#endif // _LIBFFM_INTERNAL_H
