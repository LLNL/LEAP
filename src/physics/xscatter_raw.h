#ifndef XSCATTER_RAW_H
#define XSCATTER_RAW_H

#ifdef WIN32
#pragma once
#endif

class xscatter_raw
{
public:
    xscatter_raw();
    ~xscatter_raw();
    
    static const unsigned char data[861776];
	int N_data;
};

#endif
