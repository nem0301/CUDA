#ifndef __PRINT_LOG_H__
#define __PRINT_LOG_H__

#include <stdarg.h>
#include <stdio.h>

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"

#ifndef DEBUG_OUT
#define DEBUG_OUT stdout
#endif

#ifndef DEBUG_LVL
#define DEBUG_LVL 2		// default level info level
#endif

#define DEBUG_ERR	0	// critical error
#define DEBUG_WARN	1	// warning 
#define DEBUG_INFO	2	// info as a one-time event
#define DEBUG_VERB	3	// verbose, frequent log or periodical log

#define DEBUG_MAX_LVL 3

#define DEBUG(level, fmt...) debugPrintf(level, fmt)

const char *levelColor[4] = 
{
	ANSI_COLOR_RED,
	ANSI_COLOR_YELLOW,
	ANSI_COLOR_GREEN,
	ANSI_COLOR_CYAN,
};

void debugPrintf(const int level, const char *fmt, ...)
{
	va_list pvar;
	char str[1024];

	if (level < 0 || level > DEBUG_MAX_LVL)
	{
		DEBUG(DEBUG_ERR, "[DEBUG] level error, level=%d\n", level);
		return;
	}

	if (level > DEBUG_LVL)
		return;

	va_start(pvar, fmt);
	vsprintf(str, fmt, pvar);
	fprintf(DEBUG_OUT, "%s%s" ANSI_COLOR_RESET, levelColor[level], str);
	va_end(pvar);
}

#endif
