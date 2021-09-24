mol new /home/astokely/Downloads/wisp/example_commandline/trajectory_20_frames.pdb
color Display Background white
display rendermode GLSL
display depthcue off
display projection orthographic
set molid top
set numreps [molinfo $molid get numreps]
set last_repid [expr $numreps - 1]
while {$last_repid >= 0} {
    mol delrep $last_repid top
    set update_last_repid [expr $last_repid - 1]
    set last_repid  $update_last_repid
}
mol addrep top
set molid top
set numreps [molinfo $molid get numreps]
set last_repid [expr $numreps - 1]
set repid $last_repid
mol modselect $repid top  all
set molid top
set numreps [molinfo $molid get numreps]
set last_repid [expr $numreps - 1]
set repid $last_repid
mol modcolor $repid top name
set molid top
set numreps [molinfo $molid get numreps]
set last_repid [expr $numreps - 1]
set repid $last_repid
mol modstyle $repid top QuickSurf 1.0 0.5 0.5 1
set molid top
set numreps [molinfo $molid get numreps]
set last_repid [expr $numreps - 1]
set repid $last_repid
mol modmaterial $repid top GlassBubble
