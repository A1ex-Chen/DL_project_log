import errno, os, sys

class Output():
	"""
	This class handles printing of a columed output and a CSV.
	"""

	# The table below is organized as 
	# user_option: [output_header, attribute_in_Data_class, type, min_width_in_columed_output]
	table = {
		"idx":		["Idx",			"index",	int,	7],
		"seq":		["SeqId",		"seqId",	str,	7],
		"altseq":	["AltSeqId",	"altSeqId",	str,	7],
		"tid":		["TId",			"tid",		int,	12],
		"layer":	["Layer", 		"layer",	str,	10],
		"trace":	["Trace",		"trace",	str,	25],
		"dir":		["Direction",	"dir",		str,	5],
		"sub":		["Sub",			"sub",		int,	3],
		"mod":		["Module",		"mod",		str,	15],
		"op":		["Op",			"op",		str,	15],
		"kernel":	["Kernel",		"name",		str,	0],
		"params":	["Params",		"params",	str,	0],
		"sil":		["Sil(ns)",		"sil",		int,	10],
		"tc":		["TC",			"tc",		str,	2],
		"device":	["Device",		"device",	int,	3],
		"stream":	["Stream",		"stream",	int,	3],
		"grid":		["Grid",		"grid",		str,	12],
		"block":	["Block",		"block",	str,	12],
		"flops":	["FLOPs", 		"flops",	int,	12],
		"bytes":	["Bytes",		"bytes", 	int,	12]
	}



