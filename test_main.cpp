#include <gtest/gtest.h>
#include <app.h>
int main(int argc, char **argv) {
    app::init_glog(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}