#include <cxxtest/TestSuite.h>
#include "ffm.h"
#include "ffm_internal.h"

using namespace std;
using namespace ffm;

// XXX(AR): Test failures.
class TestFmm : public CxxTest::TestSuite
{
public:
    void testReadBlockStructure(void)
    {
        ffm_block_structure* bs = ffm_read_block_structure("fixtures/block_structure.txt");
        TS_ASSERT_EQUALS(bs->nr_features, 4);
        TS_ASSERT_EQUALS(bs->max_feature, 14);
        TS_ASSERT_EQUALS(bs->max_field, 2);

        // Location of first feature
        TS_ASSERT_EQUALS(bs->index[0], 0);
        // Content
        TS_ASSERT_EQUALS(bs->features[0].f, 0);
        TS_ASSERT_EQUALS(bs->features[0].j, 10);
        TS_ASSERT_EQUALS(bs->features[0].v, 1);
        TS_ASSERT_EQUALS(bs->features[1].f, 0);
        TS_ASSERT_EQUALS(bs->features[1].j, 11);
        TS_ASSERT_EQUALS(bs->features[1].v, 1);

        // Location of next empty features
        TS_ASSERT_EQUALS(bs->index[1], 2);
        TS_ASSERT_EQUALS(bs->index[2], 2);

        // Location of next feature (and fake feature at the end)
        TS_ASSERT_EQUALS(bs->index[3], 2);
        TS_ASSERT_EQUALS(bs->index[4], 4);
        // Content
        TS_ASSERT_EQUALS(bs->features[2].f, 0);
        TS_ASSERT_EQUALS(bs->features[2].j, 13);
        TS_ASSERT_EQUALS(bs->features[2].v, 1);
        TS_ASSERT_EQUALS(bs->features[3].f, 1);
        TS_ASSERT_EQUALS(bs->features[3].j, 11);
        TS_ASSERT_EQUALS(bs->features[3].v, 1);
    }

    void testJoinFeaturesNoBlockStructure(void)
    {
        vector<ffm_node> source;
        source.push_back({0, 1, 1});
        source.push_back({0, 2, 1});
        vector<ffm_node> target;
        join_features(nullptr, source.data(), source.size(), target);
        TS_ASSERT_EQUALS(target.size(), 2);
        TS_ASSERT_EQUALS(source[0].f, target[0].f);
        TS_ASSERT_EQUALS(source[0].j, target[0].j);
        TS_ASSERT_EQUALS(source[0].v, target[0].v);
        TS_ASSERT_EQUALS(source[1].f, target[1].f);
        TS_ASSERT_EQUALS(source[1].j, target[1].j);
        TS_ASSERT_EQUALS(source[1].v, target[1].v);
    }

    void testJoinFeaturesUseBlockStructure(void)
    {
        vector<ffm_node> source;
        source.push_back({0, 0, 1});
        source.push_back({0, 2, 1});
        vector<ffm_node> target;
        ffm_block_structure* bs = ffm_read_block_structure("fixtures/block_structure.txt");
        join_features(bs, source.data(), source.size(), target);
        TS_ASSERT_EQUALS(target.size(), 4);
        TS_ASSERT_EQUALS(source[0].f, target[0].f);
        TS_ASSERT_EQUALS(source[0].j, target[0].j);
        TS_ASSERT_EQUALS(source[0].v, target[0].v);
        TS_ASSERT_EQUALS(source[1].f, target[3].f);
        TS_ASSERT_EQUALS(source[1].j, target[3].j);
        TS_ASSERT_EQUALS(source[1].v, target[3].v);
        TS_ASSERT_EQUALS(0, target[1].f);
        TS_ASSERT_EQUALS(10, target[1].j);
        TS_ASSERT_EQUALS(1, target[1].v);
        TS_ASSERT_EQUALS(0, target[2].f);
        TS_ASSERT_EQUALS(11, target[2].j);
        TS_ASSERT_EQUALS(1, target[2].v);
    }

    void testReadNegativeProbabilities(void)
    {
        ffm_int *distribution;
        distribution = read_negative_probabilities("fixtures/negative_probabilities.txt", 20);
        TS_ASSERT_EQUALS(distribution[0], 3);
        TS_ASSERT_EQUALS(distribution[5], 3);
        TS_ASSERT_EQUALS(distribution[6], 0);
        TS_ASSERT_EQUALS(distribution[19], 0);
    }

    void testReadNegativeProbabilities2(void)
    {
        ffm_int *distribution;
        distribution = read_negative_probabilities("fixtures/negative_probabilities2.txt", 10);
        TS_ASSERT_EQUALS(distribution[0], 0);
        TS_ASSERT_EQUALS(distribution[1], 0);
        TS_ASSERT_EQUALS(distribution[2], 1);
        TS_ASSERT_EQUALS(distribution[3], 3);
        TS_ASSERT_EQUALS(distribution[4], 2);
        TS_ASSERT_EQUALS(distribution[9], 2);
    }

    void testCreateNegativeSampling(void)
    {
        ffm_negative_sampling *ns = ffm_create_negative_sampling(1, 5, "fixtures/negative_probabilities2.txt", 10);
        TS_ASSERT_EQUALS(ns->num_sampling_buckets, 10);
        TS_ASSERT_EQUALS(ns->sampling_buckets[0], 0);
        TS_ASSERT_EQUALS(ns->negative_position, 1);
        TS_ASSERT_EQUALS(ns->num_negative_samples, 5);
    }
};
