from cuwisp.paths import SuboptimalPaths as Sp

sp = Sp()
sp.deserialize('out2/suboptimal_paths_0_1_2_3_4.xml')
for i in sp:
	print(i)
