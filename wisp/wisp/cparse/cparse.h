#include <string.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

void pdbFilename(
	char *pdb, 
	char *pdbPrefix, 
	int pdbIndex
);

void parsePdb(
	char *initPdbFilename, 
	char *outputPdbPrefix
);
