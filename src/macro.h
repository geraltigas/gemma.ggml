#ifndef MACROS_H
#define MACROS_H

#include <glog/logging.h>
#include <assert.h>

// define a macro to check return value and print error message, assert
#define CHECK_RT_MSG(rt, msg) if (rt != 0) { \
    LOG(ERROR) << msg; \
    assert(-1); \
}


#define CHECK_RT(rt) if (rt != 0) { \
    assert(-1); \
}

#endif //MACROS_H