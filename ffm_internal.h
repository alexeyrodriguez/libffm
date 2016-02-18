#ifndef _LIBFFM_INTERNAL_H
#define _LIBFFM_INTERNAL_H

#include "ffm.h"

using namespace std;

namespace ffm
{

inline void join_features(ffm_block_structure *bs, const vector<ffm_node> &source_features, vector<ffm_node> &target_features)
{
  target_features.clear();
  for(auto p=source_features.begin(); p<source_features.end(); p++)
  {
    target_features.push_back(*p);
    if(bs != nullptr && p->j < bs->nr_features) {
      for(int i = bs->index[p->j]; i < bs->index[p->j+1]; i++)
        target_features.push_back(bs->features[i]);
    }
  }
}

}

#endif // _LIBFFM_INTERNAL_H
