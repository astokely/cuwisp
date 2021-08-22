proc tricolor_scale {} {
  set color_start [colorinfo num]
  display update off
  for {set i 0} {$i < 1024} {incr i} {
    if {$i == 0} {
      set r 1;  set g 1;  set b 1
    }
    if {$i == 511} {
      set r 1;  set g 0;  set b 1
    }
    if {$i == 513} {
      set r 0;  set g 0;  set b 1
    }
    color change rgb [expr $i + $color_start     ] $r $g $b
  }
  display update on
}

tricolor_scale
