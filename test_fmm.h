#include <cxxtest/TestSuite.h>
#include "ffm.h"

using namespace ffm;

// XXX(AR): Test failures.
class TestFmm : public CxxTest::TestSuite
{
public:
    void testReadBlockStructure(void)
    {
        ffm_block_structure* bs = ffm_read_block_structure("fixtures/block_structure.txt");
        TS_ASSERT_EQUALS(bs->nr_features, 4);
        TS_ASSERT_EQUALS(bs->max_feature, 13);
        TS_ASSERT_EQUALS(bs->max_field, 1);

        // Location of first feature
        TS_ASSERT_EQUALS(bs->index[0], 0);
        // Content
        TS_ASSERT_EQUALS(bs->features[0].f, 0)
        TS_ASSERT_EQUALS(bs->features[0].j, 10)
        TS_ASSERT_EQUALS(bs->features[0].v, 1)
        TS_ASSERT_EQUALS(bs->features[1].f, 0)
        TS_ASSERT_EQUALS(bs->features[1].j, 11)
        TS_ASSERT_EQUALS(bs->features[1].v, 1)

        // Location of next empty features
        TS_ASSERT_EQUALS(bs->index[1], 2);
        TS_ASSERT_EQUALS(bs->index[2], 2);

        // Location of next feature (and fake feature at the end)
        TS_ASSERT_EQUALS(bs->index[3], 2);
        TS_ASSERT_EQUALS(bs->index[4], 4);
        // Content
        TS_ASSERT_EQUALS(bs->features[2].f, 0)
        TS_ASSERT_EQUALS(bs->features[2].j, 13)
        TS_ASSERT_EQUALS(bs->features[2].v, 1)
        TS_ASSERT_EQUALS(bs->features[3].f, 1)
        TS_ASSERT_EQUALS(bs->features[3].j, 11)
        TS_ASSERT_EQUALS(bs->features[3].v, 1)
    }
};
