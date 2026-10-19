// "[corridorkey] " prefixed logging through OBS's log (FR-012).
#pragma once

#include <util/base.h>

#ifdef __GNUC__
#define CK_PRINTF(a, b) __attribute__((format(printf, a, b)))
#else
#define CK_PRINTF(a, b)
#endif

namespace ck {
void log(int level, const char *fmt, ...) CK_PRINTF(2, 3);
}
