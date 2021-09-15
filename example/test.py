from cuwisp.paths import SuboptimalPaths as Sp

sp = Sp()
sp.deserialize('example_output/suboptimal_paths.xml')
for i in sp:
	print(i)
