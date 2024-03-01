#ifndef MACROS_H
#define MACROS_H

#include <glog/logging.h>
#include <assert.h>

#define CGRAPH_MAX_NODE_NUM 0

#define ON 1
#define OFF 0
#define SHOW 1
#define HIDE 0

#define CODE_MASK SHOW

#if CODE_MASK == SHOW
#define MASK(code) code
#else
#define MASK(code)
#endif

// define a macro to check return value and print error message, assert
#define CHECK_RT_MSG(rt, msg) if (rt != 0) { \
    LOG(ERROR) << msg; \
    assert(false); \
}


#define CHECK_RT(rt) if (rt != 0) { \
    assert(false); \
}

#define CHECK_PTR(ptr) if ((ptr) == nullptr) { \
    assert(false); \
}

#endif //MACROS_H