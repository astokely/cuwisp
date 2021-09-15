from cuwisp.paths import SuboptimalPaths as Sp
import os

sps = []
xmls = [f for f in os.listdir('serialized_suboptimal_paths/')]
sp = Sp()
for xml in xmls:
	sp = Sp()
	sp.deserialize('serialized_suboptimal_paths/' + xml)
	sps.append(sp)

for sp in sps:
	for p in sp:
		print(p)
	print('')

