from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

import numpy as np
from typing import Optional, Union, Tuple, List
from abserdes import Serializer as serializer
from scipy import interpolate

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
		return (
			f'{self.name} {self.bond_radius} '
			+ f'{self.sphere_resolution} '
			+ f'{self.bond_resolution}'
		)

	def __iter__(self):
		licorice_properties = {
			"name" : self.name,
			"bond_radius" : self.bond_radius,
			"sphere_resolution" : self.sphere_resolution,
			"bond_resolution" : self.bond_resolution,
		}
		for name, property_ in licorice_properties.items():
			yield name, property_

class VDW(serializer):

	def __init__(
			self,
			sphere_scale: Optional[float] = 1.0,
			sphere_resolution: Optional[float] = 12.0,
	) -> None:
		self.name = "VDW"
		self.sphere_scale = sphere_scale
		self.sphere_resolution = sphere_resolution

	def __repr__(self):
		return (
			f'{self.name} {self.sphere_scale} '
			+ f'{self.sphere_resolution}'
		)

	def __iter__(self):
		vdw_properties = {
			"name" : self.name,
			"sphere_scale" : self.sphere_scale,
			"sphere_resolution" : self.sphere_resolution,
		}
		for name, property_ in vdw_properties.items():
			yield name, property_

class Lines(serializer):

	def __init__(
			self,
			thickness: Optional[float] = 1.0,
	) -> None:
		self.name = "lines"
		self.thickness = thickness

	def __repr__(self):
		return (
			f'{self.name} {self.thickness}'
		)

	def __iter__(self):
		lines_properties = {
			"name" : self.name,
			"thickness" : self.thickness,
		}
		for name, property_ in lines_properties.items():
			yield name, property_

class QuickSurf(serializer):

	def __init__(
			self,
			radius_scale: Optional[float] = 1.0,
			density_isovalue: Optional[float] = 0.5,
			grid_spacing: Optional[float] = 1.0,
			surface_quality: Optional[int] = 1,
	):
		self.name = 'QuickSurf'
		self.radius_scale = radius_scale
		self.density_isovalue = density_isovalue
		self.grid_spacing = grid_spacing
		self.surface_quality = surface_quality

	def __repr__(self):
		return (
			f'{self.name} {self.radius_scale} '
			+ f'{self.density_isovalue} '
			+ f'{self.grid_spacing} '
			+ f'{self.surface_quality}'
		)

	def __iter__(self):
		quicksurf_properties = {
			'name' : self.name,
			'radius_scale' : self.radius_scale,
			'density_isovalue' : self.density_isovalue,
			'grid_spacing' : self.grid_spacing,
			'surface_quality' : self.surface_quality
		}
		for name, property_ in quicksurf_properties.items():
			yield name, property_


class NewCartoon(serializer):

	def __init__(
			self,
			aspect_ratio: Optional[float] = 4.10,
			thickness: Optional[float] = 0.30,
			resolution: Optional[int] = 10,
			spline_style: Optional[int] = 0,
	) -> str:
		self.name = 'NewCartoon'
		self.aspect_ratio = aspect_ratio
		self.thickness = thickness
		self.resolution = resolution
		self.spline_style = spline_style

	def __repr__(self):
		return (
			f'{self.name} {self.thickness} '
			+ f'{self.resolution} {self.aspect_ratio} '
			+ f'{self.spline_style}'
		)

	def __iter__(self):
		new_cartoon_properties = {
			'name' : self.name,
			'thickness' : self.thickness,
			'resolution' : self.resolution,
			'aspect_ratio' : self.aspect_ratio,
			'spline_style' : self.spline_style
		}
		for name, property_ in new_cartoon_properties.items():
			yield name, property_

class Sphere(serializer):

	def __init__(
			self,
			center: np.ndarray,
			radius: Optional[float] = 1.0,
			resolution: Optional[int] = 10,
	) -> None:
		self.name = 'sphere'
		self.center = center
		self.radius = radius
		self.resolution = resolution

	def __repr__(self):
		x, y, z = self.center
		return (
			f'draw sphere {{{x} {y} {z}}} '
			+ f'radius {self.radius} '
			+ f'resolution {self.resolution}'
		)

	def __iter__(self):
		sphere_properties = {
			"name" : self.name,
			"center" : self.center,
			"radius" : self.radius,
			"resolution" : self.resolution,
		}
		for name, property_ in sphere_properties.items():
			yield name, property_

	def tcl(
			self,
			tcl: Optional[str] = '',
			new_line: Optional = True,
	) -> str:
		if new_line:
			tcl += ''.join([self.__repr__(), '\n'])
		else:
			tcl += self.__repr__()
		return tcl

class Cylinder(serializer):

	def __init__(
			self,
			center1: np.ndarray,
			center2: np.ndarray,
			radius: Optional[float] = 0.3,
			resolution: Optional[int] = 10,
			filled: Optional[int] = 0,
	) -> None:
		self.name = 'cylinder'
		self.center1 = center1
		self.center2 = center2
		self.radius = radius
		self.resolution = resolution
		self.filled = filled

	def __repr__(self):
		x1, y1, z1 = self.center1
		x2, y2, z2 = self.center2
		return (
			f'draw cylinder '
			+ f'{{{x1} {y1} {z1}}} {{{x2} {y2} {z2}}} '
			+ f'radius {self.radius} '
			+ f'resolution {self.resolution} '
			+ f'filled {self.filled}'
		)

	def __iter__(self):
		cylinder_properties = {
			"name" : self.name,
			"center1" : self.center1,
			"center2" : self.center2,
			"radius" : self.radius,
			"resolution" : self.resolution,
			"filled" : self.filled,
		}
		for name, property_ in cylinder_properties.items():
			yield name, property_

	def tcl(
			self,
			tcl: Optional[str] = '',
			new_line: Optional[bool] = True,
	) -> str:
		if new_line:
			tcl += ''.join([self.__repr__(), '\n'])
		else:
			tcl += self.__repr__()
		return tcl

class Material(serializer):

	def __init__(
		self,
		name: str,
		ambient: Optional[float] = 0.0,
		diffuse: Optional[float] = 0.65,
		specular: Optional[float] = 0.50,
		shininess: Optional[float] = 0.53,
		mirror: Optional[float] = 0.0,
		opacity: Optional[float] = 1.0,
		outline: Optional[float] = 0.0,
		outline_width: Optional[float] = 0.0,
		angle_modulated_transparency: Optional[bool] = False,
	) -> None:
		self.name = name
		self.ambient = ambient
		self.diffuse = diffuse
		self.specular = specular
		self.shininess = shininess
		self.mirror = mirror
		self.opacity = opacity
		self.outline = outline
		self.outline_width = outline_width
		self.angle_modulated_transparency = (
			angle_modulated_transparency
		)

	def __repr__(self):
		return (
			f'material change ambient {self.name} {self.ambient}\n'
			+ f'material change diffuse {self.name} {self.diffuse}\n'
			+ f'material change specular {self.name} {self.specular}\n'
			+ f'material change mirror {self.name} {self.mirror}\n'
			+ f'material change opacity {self.name} {self.opacity}\n'
			+ f'material change outline {self.name} {self.outline}\n'
			+ f'material change outlinewidth {self.name} '
			+ f'{self.outline_width}\n'
			+ f'material change transmode {self.name} '
			+ f'{int(self.angle_modulated_transparency)}\n'
		)

	def __iter__(self):
		material_properties = {
			'name' : self.name,
			'diffuse' : self.diffuse,
			'specular' : self.specular,
			'mirror' : self.mirror,
			'opacity' : self.opacity,
			'outline' : self.outline,
			'outlinewidth' : self.outline_width
		}
		for name, property_ in material_properties.items():
			yield name, property_


def generate_spline(
		nodes: np.ndarray,
		spline_input_points_incr: Optional[float] = 0.001,
		smoothing_factor: Optional[float] = 0.0,
		column_major: Optional[bool] = False,
) -> np.ndarray:
	num_edges = (
		max(nodes.shape) - 1
	)
	degree = num_edges - 1
	if num_edges > 3:
		degree = 3
	if not column_major:
		nodes = nodes.T
	x, y, z = nodes
	tck, u = interpolate.splprep(
		[x, y, z],
		s=smoothing_factor,
		k=degree
	)
	u_new = np.arange(
		0,
		1.0 + spline_input_points_incr,
		spline_input_points_incr
	)
	return interpolate.splev(u_new, tck)

def draw_curve_from_spline(
		spline: np.ndarray,
		color: Union[Tuple, str, int] = 0,
		molid: Optional[int] = 'top',
		radius: Optional[float] = 0.3,
		resolution: Optional[int] = 100,
		filled: Optional[int] = 0,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	if isinstance(color, tuple):
		color_gradient, index = color
		tcl += (
			f'set color_index [lindex '
			+ f'${color_gradient} {index}]\n'
			+ f'graphics {molid} color $color_index]\n'
		)
	else:
		tcl += set_graphics_color(color, molid=molid)
	for i in range(len(spline[0]) - 1):
		c1 = np.array([
			spline[0][i],
			spline[1][i],
			spline[2][i]
		])
		c2 = np.array([
			spline[0][i+1],
			spline[1][i+1],
			spline[2][i+1]
		])
		cylinder = Cylinder(
			c1,
			c2,
			radius=radius,
			resolution=resolution,
			filled=filled
		)
		tcl += cylinder.tcl()
	if new_line:
		return tcl
	return tcl[:-1]

def reset_view(
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'display resetview\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_projection(
		projection: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'display projection {projection}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_depth_cueing(
		depth_cueing: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'display depthcue {depth_cueing}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def load_pdb(
		pdb: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'mol new {pdb}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def delete_molecule(
		molid: Optional[int] = 'top',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'mol delete {molid}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_antialiasing(
		antialiasing: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'display antialias {antialiasing}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_render_mode(
		render_mode: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'display rendermode {render_mode}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_background_color(
		colour: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'color Display Background {colour}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def delete_material(
		material_name: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'set avail_materials [material list]\n'
	tcl += (
		f'if {{[lsearch -exact $avail_materials "'
		f'{material_name}"] >= 0}} {{\n'
		f'	  material delete {material_name}\n'
		f'}}\n'
	)
	if new_line:
		return tcl
	return tcl[:-1]

def add_material(
		material: Material,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += delete_material(material.name)
	tcl += f'material add {material.name}\n'
	material_copy = str(material)
	tcl = ''.join([tcl, material_copy])
	if new_line:
		return tcl
	return tcl[:-1]

def last_color_index(
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	if new_line:
		tcl += f'set last_color_index [expr [colorinfo num] - 1] \n'
	else:
		tcl += f'set last_color_index [expr [colorinfo num] - 1]'
	return tcl

def new_color_index(
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'set color_index [colorinfo num] \n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_representation(
			representation: Licorice,
			tcl: Optional[str] = '',
			new_line: Optional[bool] = True
):
	tcl += f'mol representation {representation}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def modify_representation(
		representation: Licorice,
		molid: Optional[int] = 'top',
		repid: Optional[int] = None,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	if repid is None:
		tcl += get_last_repid()
		tcl += f'set repid $last_repid\n'
	else:
		tcl += f'set repid {repid}\n'
	tcl += f'mol modstyle $repid {molid} {representation}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def add_representation(
		molid: Optional[int] = 'top',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'mol addrep {molid}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def get_last_repid(
		molid: Optional[int] = 'top',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True
) -> str:
	tcl += f'set molid {molid}\n'
	tcl += f'set numreps [molinfo $molid get numreps]\n'
	tcl += f'set last_repid [expr $numreps - 1]\n'
	if new_line:
		return tcl
	return tcl[:-1]

def add_color(
		red: float,
		green: float,
		blue: float,
		color_index: Optional[int] = None,
		incr: Optional[int] = 0,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True
) -> str:
	if color_index is None:
		tcl += new_color_index()
		tcl += f'set colour_index [expr $color_index + {incr}]\n'
	else:
		tcl += f'set colour_index {color_index}\n'
	tcl += f'color change rgb $colour_index {red} {green} {blue}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_graphics_color(
		colour: Union[str, int],
		molid: Optional[int] = 'top',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'graphics {molid} color {colour}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_draw_color(
		colour: Union[str, int],
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'draw color {colour}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_draw_material(
		material: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'draw material {material}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def set_representation_coloring_method(
		coloring_method: Union[str, Tuple],
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	if isinstance(coloring_method, tuple):
		coloring_method = tuple_list_ndarray_to_str(
			coloring_method
		)
	tcl += f'mol color {coloring_method}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def modify_representation_coloring_method(
		coloring_method: Union[str, Tuple],
		molid: Optional[int] = 'top',
		repid: Optional[int] = None,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	if isinstance(coloring_method, tuple):
		coloring_method = tuple_list_ndarray_to_str(
			coloring_method
		)
	if repid is None:
		tcl += get_last_repid()
		tcl += f'set repid $last_repid\n'
	else:
		tcl += f'set repid {repid}\n'
	tcl += f'mol modcolor $repid {molid} {coloring_method}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def delete_all_representations(
		molid: Optional[int] = 'top',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += get_last_repid(molid=molid)
	tcl += (
		f'while {{$last_repid >= 0}} {{\n'
		+ f'	mol delrep $last_repid {molid}\n'
		+ f'	set update_last_repid [expr $last_repid - 1]\n'
		+ f'	set last_repid	$update_last_repid\n'
		+ f'}}\n'
	)
	if new_line:
		return tcl
	return tcl[:-1]

def tuple_list_ndarray_to_str(
	tuple_or_list_or_array: Union[List, np.ndarray],
) -> str:
	s = ''
	for val in tuple_or_list_or_array:
		s += f'{val} '
	return s[:-1]

def modify_representation_selection(
		selection: Union[str, Tuple],
		molid: Optional[int] = 'top',
		repid: Optional[int] = None,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True
) -> str:
	if isinstance(selection, tuple):
		if isinstance(selection[1], str):
			selection_key, selection_value = selection
			selection_value = ''.join(['$', selection_value])
		else:
			selection_key, selection_value = selection
			selection_value = tuple_list_ndarray_to_str(selection_value)
	else:
		selection_key = ''
		selection_value = selection
	if repid is None:
		tcl += get_last_repid()
		tcl += f'set repid $last_repid\n'
	else:
		tcl += f'set repid {repid}\n'
	tcl += (
		f'mol modselect $repid {molid} '
		+ f'{selection_key} {selection_value}\n'
	)
	if new_line:
		return tcl
	return tcl[:-1]


def atomselect(
		name: str,
		selection: Union[str, Tuple],
		molid: Optional[int] = 'top',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
):
	if isinstance(selection, tuple):
		selection_key, selection_value = selection
		selection_value = tuple_list_ndarray_to_str(selection_value)
		selection = ''.join([selection_key, ' ', selection_value])
	tcl += f'set {name} [atomselect {molid} "{selection}"]\n'
	if new_line:
		return tcl
	return tcl[:-1]

def selection_atom_indices(
		name: str,
		selection_name: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
):
	tcl += f'set {name} [${selection_name} list]\n'
	if new_line:
		return tcl
	return tcl[:-1]

def modify_representation_material(
		material_name: str,
		molid: Optional[int] = 'top',
		repid: Optional[int] = None,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	if repid is None:
		tcl += get_last_repid()
		tcl += f'set repid $last_repid\n'
	else:
		tcl += f'set repid {repid}\n'
	tcl += f'mol modmaterial $repid {molid} {material_name}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def pbc_wrap(
		center_selection: str,
		center: Optional[str] = 'com',
		compound: Optional[str] = 'residue',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += (
		f'pbc wrap -center {center} '
		+ f'-centersel "{center_selection}" '
		+ f'-compound {compound} -all\n'
	)
	if new_line:
		return tcl
	return tcl[:-1]


def modify_representation_style(
		style_name: str,
		molid: Optional[int] = 'top',
		repid: Optional[int] = None,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	if repid is None:
		tcl += get_last_repid()
		tcl += f'set repid $last_repid\n'
	else:
		tcl += f'set repid {repid}\n'
	tcl += f'mol modstyle $repid {molid} {style_name}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def get_center_of_mass(
		selection_name: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'measure center ${selection_name} weight mass\n'
	if new_line:
		return tcl
	return tcl[:-1]

def center_of_mass_selection(
		name: str,
		selection_name: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	com = get_center_of_mass(
		selection_name,
		new_line=False
	)
	tcl += f'set {name} [{com}]\n'
	if new_line:
		return tcl
	return tcl[:-1]

def move_selection_to_origin_transformation(
		name: str,
		selection_name: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	com = get_center_of_mass(
		selection_name,
		new_line=False,
	)
	tcl += f'set {name} [vecsub {{0 0 0}} [{com}]]\n'
	if new_line:
		return tcl
	return tcl[:-1]

def apply_transformation_to_selection(
		selection_name: str,
		transformation_name: str,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'${selection_name} moveby ${transformation_name}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def create_color_gradient(
		name: str,
		colors: List,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	tcl += f'set list {name}\n'
	tcl += new_color_index()
	index = 0
	for color in colors:
		red, green, blue = color
		tcl += add_color(red, green, blue, incr=index)
		tcl += f'lappend {name} [expr $color_index + {index}]\n'
		index += 1
	if new_line:
		return tcl
	return tcl[:-1]

def tcl_proc(
		proc_name: str,
		tcl: str,
		new_line: Optional[bool] = True,
) -> str:
	proc = f'proc {proc_name} {{}} {{\n'
	proc += f'{tcl}'
	if proc[-1] != '\n':
		proc += f'\n'
	proc += '}\n'
	tcl = proc
	if new_line:
		return tcl
	return tcl[:-1]




def save_tcl(
		tcl: str,
		tcl_filename: str,
		write_method: Optional[str] = 'w',
) -> None:
	with open(tcl_filename, write_method) as tcl_file:
		tcl_file.write(tcl)
		tcl_file.close()
