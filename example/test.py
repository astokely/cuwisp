from cuwisp.paths import SuboptimalPaths as Sp

sp = Sp()
sp.deserialize('example_output/suboptimal_paths_0_1_2_3_4_5.xml')
for i in sp:
	print(i)
