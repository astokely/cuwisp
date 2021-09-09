/*
By Andy Stokely
*/
#include "cparse.h"

void pdbFilename(
	char *pdb, 
	char *pdbPrefix, 
	int frame 
	) {
	snprintf(
		pdb, 1000, "%s%d.pdb", 
		pdbPrefix, frame 
	);
}

void parsePdb(
		char *initPdbFilename, 
		char *outputPdbPrefix
	) {
	FILE* initPdb = fopen(initPdbFilename, "r"); 
	char initPdbBuffer[80];
	int frame = 0;
	char tmpPdb[1000];
 	pdbFilename(tmpPdb, outputPdbPrefix, frame);
	FILE *outputPdb = fopen(tmpPdb, "w");
	char cryst[80];
	int firstIter = 1;
	int hasCrystHeader = 0;
	int openNewPdb = 0;
    while (fgets(initPdbBuffer, sizeof(initPdbBuffer), initPdb)) {
		if (firstIter == 1) {
			char *firstLine;
			firstLine = strstr(initPdbBuffer, "CRYST");
			if (firstLine) {
				strcpy(cryst, initPdbBuffer);
				hasCrystHeader = 1;
			}
			firstIter = 0;
		}
		char *end;
		end = strstr(initPdbBuffer, "END");
		if (end) {
			fputs(initPdbBuffer, outputPdb);
			fclose(outputPdb);
			frame++;
			openNewPdb = 1;
		}
		else {
			if (openNewPdb == 1) {
				pdbFilename(tmpPdb, outputPdbPrefix, frame);
				outputPdb = fopen(tmpPdb, "w");
				if (hasCrystHeader) {
					fputs(cryst, outputPdb);
				}
			fputs(initPdbBuffer, outputPdb);
			openNewPdb = 0;
		}
			else {
				fputs(initPdbBuffer, outputPdb);
			}
		}
	}
	fclose(initPdb);
//	fclose(outputPdb);
}

