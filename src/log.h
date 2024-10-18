#ifndef __LOG_H__
#define __LOG_H__

#ifdef WIN32
#pragma once
#endif

#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <string>
#include <stdio.h>
#include <iomanip>

//inline std::string NowTime();

/**
 * This file provides a header-only implementation of a logging utility.
 */

enum TLogLevel {logERROR, logWARNING, logINFO, logSTATUS, logDEBUG, logDEBUG1, logDEBUG2, logDEBUG3, logDEBUG4};

//*
class Log
{
public:
    Log();
    virtual ~Log();
    std::ostringstream& Get(TLogLevel level, std::string class_name, std::string func_name);
public:
    static TLogLevel& ReportingLevel();
    static std::string ToString(TLogLevel level);
    static TLogLevel FromString(const std::string& level);
    static std::ofstream*& Stream();
protected:
    std::ostringstream os;
    TLogLevel log_level;
private:
    //static string className;
    Log(const Log&);
    Log& operator =(const Log&);
};

inline Log::Log()
{
	//className = "Log";
}

inline std::ostringstream& Log::Get(TLogLevel level, std::string class_name, std::string func_name)
{
//    os << "- " << NowTime();
//    os << " " << ToString(level) << ": ";
    os << std::string(level > logDEBUG ? level - logDEBUG : 0, '\t');
    if((level >= logDEBUG) || (level == logERROR) || (level == logWARNING))
    	os << class_name << "::" << func_name << " - ";

    log_level = level;
    return os;
}

inline Log::~Log()
{
    //os << std::endl;
    //fprintf(stderr, "%s", os.str().c_str());
    //fflush(stderr);

    std::cout << os.str() << std::flush;

    std::ofstream* pStream = Stream();
    if ((pStream != NULL) && (log_level != logSTATUS))
    //if ((pStream != NULL))
    	*pStream << os.str()<< std::flush;
}

inline TLogLevel& Log::ReportingLevel()
{
    //static TLogLevel reportingLevel = logSTATUS;
    static TLogLevel reportingLevel = logWARNING;
    return reportingLevel;
}

inline std::string Log::ToString(TLogLevel level)
{
	static const char* const buffer[] = {"ERROR", "WARNING", "INFO", "STATUS", "DEBUG", "DEBUG1", "DEBUG2", "DEBUG3", "DEBUG4"};
    return buffer[level];
}

inline TLogLevel Log::FromString(const std::string& level)
{
    if (level == "DEBUG4")
        return logDEBUG4;
    if (level == "DEBUG3")
        return logDEBUG3;
    if (level == "DEBUG2")
        return logDEBUG2;
    if (level == "DEBUG1")
        return logDEBUG1;
    if (level == "DEBUG")
        return logDEBUG;
    if (level == "STATUS")
        return logSTATUS;
    if (level == "INFO")
        return logINFO;
    if (level == "WARNING")
        return logWARNING;
    if (level == "ERROR")
        return logERROR;
    //Log().Get(logWARNING,className,"FromString") << "Unknown logging level '" << level << "'. Using STATUS level as default." << endl;
    Log().Get(logWARNING,"Log","FromString") << "Unknown logging level '" << level << "'. Using STATUS level as default." << std::endl;
    return logSTATUS;
}

inline std::ofstream*& Log::Stream()
{
    static std::ofstream* pStream = NULL;
    return pStream;
}


#define LOG(level,class_name,func_name) \
    if (level > Log::ReportingLevel()) ; \
    else Log().Get(level,class_name,func_name)

//*/

/*
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)

#include <windows.h>

inline std::string NowTime()
{
    const int MAX_LEN = 200;
    char buffer[MAX_LEN];
    if (GetTimeFormatA(LOCALE_USER_DEFAULT, 0, 0, 
            "HH':'mm':'ss", buffer, MAX_LEN) == 0)
        return "Error in NowTime()";

    char result[100] = {0};
    static DWORD first = GetTickCount();
    sprintf(result, "%s.%03ld", buffer, (long)(GetTickCount() - first) % 1000);
    return result;
}

#else

#include <sys/time.h>

inline std::string NowTime()
{
    char buffer[11];
    time_t t;
    time(&t);
    tm r = {0};
    strftime(buffer, sizeof(buffer), "%X", localtime_r(&t, &r));
    struct timeval tv;
    gettimeofday(&tv, 0);
    char result[100] = {0};
    std::sprintf(result, "%s.%03ld", buffer, (long)tv.tv_usec / 1000); 
    return result;
}

#endif //WIN32
//*/

#endif //__LOG_H__
