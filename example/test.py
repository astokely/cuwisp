from cuwisp.paths import SuboptimalPaths as sp

p = sp()
p.deserialize("example_output/suboptimal_paths.xml")
print(p)
