import sys

import genotypes
from graphviz import Digraph




if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]

    try:
        genotype = eval("genotypes.{}".format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name))
        sys.exit(1)

    plot(genotype.normal, "normal")
    plot(genotype.reduce, "reduction")