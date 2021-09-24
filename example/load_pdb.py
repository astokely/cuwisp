from cuwisp.visualize import VmdRepresentation, load_pdb
from cuwisp.vmdtcl import save_tcl, QuickSurf

tcl = ''
vmd_init_rep = VmdRepresentation(
	style=QuickSurf(grid_spacing=0.5),
	material='GlassBubble',
)
tcl = load_pdb(
	'/home/astokely/Downloads/wisp/example_commandline/trajectory_20_frames.pdb', 
	initial_representation=vmd_init_rep,
	tcl=tcl,
)
save_tcl(tcl, 'load_pdb.tcl')
