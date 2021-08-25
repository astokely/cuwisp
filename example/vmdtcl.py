from typing import Optional
from abserdes import Serializer as serializer
import os
def delete_material(
		material_name: str
) -> str:
	tcl = f'set avail_materials [material list]\n' 
	tcl += (
		f'if {{[lsearch -exact $avail_materials "{material_name}"] >= 0}} {{\n'
		f'    material delete {material_name}\n'
		f'}}\n'
	)
	return tcl

def add_material(
		name: str,
		properties: Optional[Dict] = {},
) -> str:
	tcl = delete_material(name)
	tcl += f'material add {name}\n' 
	for property_name, property_ in properties.items():
		tcl += (
			f'material change {property_name} {name} {property_}\n'
		)
	return tcl

def new_color_index() -> str:
	tcl = f'set color_start [colorinfo num] \n'
	return tcl

class Licorice(serializer):

	def __init__(
			self,
			sphere_resolution: Optional[int] = 12, 
			bond_radius: Optional[int] = 0.3, 
			bond_resolution: Optional[int] = 12, 
	) -> None:
		self.name = "Licorice"
		self.sphere_resolution = sphere_resolution
		self.bond_radius = bond_radius
		self.bond_resolution = bond_resolution

	def __repr__(self):
		name = self.name
		sphere_resolution = self.sphere_resolution
		bond_radius = self.bond_radius
		bond_resolution = self.bond_resolution
		return (
			f'{name} {sphere_resolution} '
			f'{bond_radius} {bond_resolution}'
		)

	def __iter__(self):
		licorice_properties = {
			"name" : self.name,
			"sphere_resolution" : self.sphere_resolution,
			"bond_radius" : self.bond_radius,
			"bond_resolution" : self.bond_resolution,
		}
		for name, property_ in licorice_properties.items():
			yield name, property_

def set_representation(representation, molid, new_line = True):
	if new_line:
		tcl = f'mol representation {molid} {representation}\n'
	else:
		tcl = f'mol representation {molid} {representation}'
	return tcl

def modify_representation(representation, molid, repid, new_line = True):
	if new_line:
		tcl = f'mol modstyle {repid} {molid} {representation}\n'
	else:
		tcl = f'mol modstyle {repid} {molid} {representation}'
	return tcl

def get_top_molid(new_line = True):
	if new_line:
		tcl = f'molinfo top\n'
	else:
		tcl = f'molinfo top'
	return tcl

def get_last_repid(molid=None, new_line = True):
	if molid == None:
		molid = get_top_molid(False)
	tcl = f'set molid [{molid}]\n'
	tcl += f'set numreps [molinfo $molid get numreps]\n'
	if new_line:
		tcl += f'set last_repid [expr $numreps - 1]\n'
	else:
		tcl += f'set last_repid [expr $numreps - 1]'
	return tcl

properties = {
	"ambient" : 0.0,
	"specular" : 0.0,
	"diffuse" : 0.79,
	"shininess" : 0.53,
	"mirror" : 0.0,
	"opacity" : 1.0
}

name = "new_node"

licorice = Licorice(sphere_resolution=250)	
#print(modify_representation(licorice, 5, 5, False))
print(get_last_repid())
