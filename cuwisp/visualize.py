from colour import Color
from typing import Optional, Tuple, Union, NamedTuple, Dict, List
from cuwisp.paths import SuboptimalPaths
from cuwisp.paths import Path
from cuwisp import vmdtcl
import numpy as np
from abserdes import Serializer as serializer
import os

class VisualizeSuboptimalPaths(serializer):

	def __init__(
			self,
			suboptimal_paths: SuboptimalPaths,
			color: Union[str, Tuple[str]],
			radii: Union[float, Tuple[float]],
			node_spheres: Optional[Dict] = False,
			src_node_sphere: Optional[Dict] = False,	
			sink_node_sphere: Optional[Dict] = False,	
			node_atoms_representation: Optional[Dict] = False,
	) -> None:
		self.suboptimal_paths = suboptimal_paths
		self.node_spheres = node_spheres
		self.src_node_sphere = src_node_sphere
		self.sink_node_sphere = sink_node_sphere
		self.node_atoms_representation = node_atoms_representation
		self.radii = radii
		self.color = color 

class VmdRepresentation(serializer):

	def __init__(
			self,
			selection: Optional[Union[str, Tuple]] = 'all',
			coloring_method: Optional[
				Union[str, Tuple]
			] = 'name',
			style: Optional[Union[
				vmdtcl.Lines,
				vmdtcl.VDW,
				vmdtcl.Licorice,
				vmdtcl.QuickSurf,
			]] = vmdtcl.Lines(),
			material: Optional[str] = 'Opaque',
	) -> None:
		self.selection = selection
		self.coloring_method = coloring_method
		self.style = style
		self.material = material

	def __repr__(self):
		tcl = ''
		tcl = vmdtcl.add_representation(tcl=tcl)
		tcl = vmdtcl.modify_representation_selection(
			self.selection, 
			tcl=tcl
		)
		tcl = vmdtcl.modify_representation_coloring_method(
			self.coloring_method,
			tcl=tcl
		)
		tcl = vmdtcl.modify_representation_style(
			self.style,
			tcl=tcl
		)
		tcl = vmdtcl.modify_representation_material(
			self.material,
			tcl=tcl
		)
		return tcl

		
def get_src_and_sink_node_coordinates(
		path: Union[
			SuboptimalPaths, Path
		],
) -> Tuple[np.ndarray]:
	if isinstance(path, SuboptimalPaths):
		src_coordinates = path[0][0][0].coordinates
		sink_coordinates = path[0][-1][1].coordinates
	else:
		src_coordinates = path[0][0].coordinates
		sink_coordinates = path[-1][1].coordinates
	return (
		src_coordinates,
		sink_coordinates
	)

def get_node_coordinates(
		path: Path,
) -> Tuple[np.ndarray]:
	src_coordinates, sink_coordinates = (
		get_src_and_sink_node_coordinates(path)
	)
	coordinates = [src_coordinates]
	for edge in path[1:]:
		coordinates.append(edge.node1.coordinates)
	coordinates.append(sink_coordinates)
	return coordinates

def get_color_gradient(
		color1: str,
		color2: str,
		num_colors: int,
) -> List:
	color1 = Color(color1)
	color2 = Color(color2)
	colors = list(color1.range_to(color2, num_colors))
	return [
		color.rgb for color in colors
	] 

def get_radii(
		smallest: float,
		largest: float,
		num_radii: int,
		reverse: Optional[bool] = True,
) -> np.ndarray:
	if reverse:
		return np.linspace(
			largest,
			smallest,
			num_radii
		)
	return np.linspace(
		smallest,
		largest,
		num_radii
	)


def draw_suboptimal_paths(
		parameters: VisualizeSuboptimalPaths,
		tcl_output_filename: Optional[str] = (
			'suboptimal_paths_curve.tcl'
		),
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	suboptimal_paths = parameters.suboptimal_paths
	num_paths = len(suboptimal_paths)
	src_node_coordinates, sink_node_coordinates = (
		get_src_and_sink_node_coordinates(suboptimal_paths)
	)	
	if parameters.src_node_sphere:
		src_node_sphere_color = (
			parameters.src_node_sphere.pop('color')
		)
		tcl = vmdtcl.set_draw_color(
			src_node_sphere_color,
			tcl=tcl
		)
		parameters.src_node_sphere['center'] = (
			src_node_coordinates 
		)
		src_node_sphere = vmdtcl.Sphere(
			**parameters.src_node_sphere
		)
		tcl = src_node_sphere.tcl(tcl=tcl)
	if parameters.sink_node_sphere:
		sink_node_sphere_color = (
			parameters.sink_node_sphere.pop('color')
		)
		parameters.sink_node_sphere['center'] = (
			sink_node_coordinates 
		)
		tcl = vmdtcl.set_draw_color(
			sink_node_sphere_color,
			tcl=tcl
		)
		sink_node_sphere = vmdtcl.Sphere(
			**parameters.sink_node_sphere
		)
		tcl = sink_node_sphere.tcl(tcl=tcl)
	if isinstance(parameters.radii, tuple):
		radii = get_radii(*parameters.radii, num_paths)
	else:
		radii = [
			parameters.radii for path
			in range(num_paths)
		]
	if isinstance(parameters.color, str):
		color_gradient = get_color_gradient(
			parameters.color,
			parameters.color,
			num_paths
		)
	else:
		color_gradient = get_color_gradient(
			*parameters.color,
			num_paths
		)
	tcl = vmdtcl.create_color_gradient(
		'color_gradient',
		color_gradient,
		tcl=tcl
	)
	color_and_radii_index = 0
	if parameters.node_spheres:
		node_spheres_color = (
			parameters.node_spheres.pop('color')
		)
	for path in suboptimal_paths:
		node_coordinates = get_node_coordinates(path)
		node_index = 0
		for node_coordinate in node_coordinates[1:-1]:
			if parameters.node_spheres:
				parameters.node_spheres['center'] = node_coordinate
				node_sphere = vmdtcl.Sphere(
					**parameters.node_spheres
				)
				tcl = vmdtcl.set_draw_color(
					node_spheres_color, 
					tcl=tcl
				)
				tcl = node_sphere.tcl(tcl=tcl)
			node_index += 1
		spline = vmdtcl.generate_spline(np.array(node_coordinates))
		tcl = vmdtcl.draw_curve_from_spline(
			spline, color=(
				'color_gradient', 
				color_and_radii_index
			), 
			radius=radii[color_and_radii_index], 
			tcl=tcl
		)
		color_and_radii_index += 1
	if new_line:
		return tcl
	return tcl[:-1]

def load_pdb(
		pdb_filename,
		background_color: Optional[Union[str, int]] = 'white',
		render_mode: Optional[str] = 'GLSL',
		depth_cueing: Optional[str] = 'off',
		projection: Optional[str] = 'orthographic',
		initial_representation: Optional[VmdRepresentation] = None,
		tcl: Optional[str] = '',
		new_line: Optional[str] = True,
) -> str:
	tcl = vmdtcl.load_pdb(pdb_filename, tcl=tcl)
	tcl = vmdtcl.set_background_color(background_color, tcl=tcl)	
	tcl = vmdtcl.set_render_mode(render_mode, tcl=tcl)	
	tcl = vmdtcl.set_depth_cueing(depth_cueing, tcl=tcl)	
	tcl = vmdtcl.set_projection(projection, tcl=tcl)	
	tcl = vmdtcl.delete_all_representations(tcl=tcl)
	if initial_representation != None:
		initial_representation_copy = (
			str(initial_representation)
		)
		tcl = ''.join([tcl, initial_representation_copy]) 
	if new_line:
		return tcl
	return tcl[:-1]
	
	


