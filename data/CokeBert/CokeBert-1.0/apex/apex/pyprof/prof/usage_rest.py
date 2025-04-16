import sys
import argparse



	parser = argparse.ArgumentParser(prog=sys.argv[0], description="PyTorch Profiler", formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument("file",
		nargs='?',
		type=str,
		default=None,
		help="Output of parse.py (Python dictionary).")

	parser.add_argument("-c",
		type=check_cols,
		default="idx,dir,sub,mod,op,kernel,params,sil",
		help='''Comma seperated names of columns to print.
idx:      Index
seq:      PyTorch Sequence Id
altseq:   PyTorch Alternate Sequence Id
tid:      Thread Id
layer:    User annotated NVTX string (can be nested)
trace:    Function Call Trace
dir:      Direction
sub:      Sub Sequence Id
mod:      Module
op:       Operattion
kernel:   Kernel Name
params:   Parameters
sil:      Silicon Time (in ns)
tc:       Tensor Core Usage
device:   GPU Device Id
stream:   Stream Id
grid:     Grid Dimensions
block:    Block Dimensions
flops:    Floating point ops (FMA = 2 FLOPs)
bytes:    Number of bytes in and out of DRAM
e.g. -c idx,kernel,sil''')

	group = parser.add_mutually_exclusive_group()
	group.add_argument("--csv",
		action="store_true",
		default=False,
		help="Print a CSV output.")
	group.add_argument("-w",
		type=int,
		default=0,
		help="Width of columnated output.")

	args = parser.parse_args()
	if args.file is None:
		args.file = sys.stdin
	else:
		args.file = openFile(args.file)
	return args