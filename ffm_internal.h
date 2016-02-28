#ifndef _LIBFFM_INTERNAL_H
#define _LIBFFM_INTERNAL_H

#include <map>
#include "ffm.h"

using namespace std;

namespace ffm
{

inline void join_features(ffm_block_structure *bs, ffm_node *begin, ffm_int size, vector<ffm_node> &target_features)
{
  target_features.clear();
  for(ffm_int i=0; i<size; i++)
  {
    ffm_node *p = begin + i;
    target_features.push_back(*p);
    if(bs != nullptr && p->j < bs->nr_features) {
      for(int i = bs->index[p->j]; i < bs->index[p->j+1]; i++)
        target_features.push_back(bs->features[i]);
    }
  }
}

ffm_int *read_negative_probabilities(char const *path, ffm_int n);

}

#endif // _LIBFFM_INTERNAL_H
