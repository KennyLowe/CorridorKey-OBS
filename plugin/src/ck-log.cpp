#include "ck-log.hpp"

#include <cstdarg>
#include <cstdio>
#include <string>

namespace ck {

void log(int level, const char *fmt, ...)
{
	char buf[2048];
	va_list ap;
	va_start(ap, fmt);
	vsnprintf(buf, sizeof(buf), fmt, ap);
	va_end(ap);
	blog(level, "[corridorkey] %s", buf);
}

} // namespace ck
