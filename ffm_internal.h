#ifndef _LIBFFM_INTERNAL_H
#define _LIBFFM_INTERNAL_H

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

inline void join_features(ffm_block_structure *bs, ffm_node *begin, ffm_int size, vector<ffm_node> &target_features)
{
    join_features(nullptr, false, nullptr, bs, begin, size, target_features);
}

ffm_int *read_negative_probabilities(char const *path, ffm_int n);

}

#endif // _LIBFFM_INTERNAL_H
