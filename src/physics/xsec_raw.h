#ifndef XSEC_RAW_H
#define XSEC_RAW_H

#ifdef WIN32
#pragma once
#endif

class xsec_raw
{
public:
	xsec_raw();
	~xsec_raw();

	static const unsigned char data[725144];
	int N_data;
};

#endif
