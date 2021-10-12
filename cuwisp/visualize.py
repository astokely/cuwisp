from __future__ import absolute_import

__author__ = "Andy Stokely"
__version__ = "1.0"

from colour import Color
from typing import Optional, Tuple, Union, NamedTuple, Dict, List
from cuwisp.paths import SuboptimalPaths
from cuwisp.paths import Path
from cuwisp.nodes import Nodes 
from cuwisp.vmdtcl import QuickSurf 
import numpy as np
import cuwisp.vmdtcl as vmdtcl
from abserdes import Serializer as serializer
import os

class VisualizeSuboptimalPaths(serializer):

	def __init__(
			self,
			suboptimal_paths: SuboptimalPaths,
			color: Union[str, Tuple[str]],
			radii: Union[float, Tuple[float]],
			frame: Optional[int] = 0,
			node_spheres: Optional[Dict] = False,
			src_node_sphere: Optional[Dict] = False,	
			sink_node_sphere: Optional[Dict] = False,	
			node_atom_representations: Optional[Dict] = False,
			path_indices: Optional[List[int]] = False,
	) -> None:
		self.suboptimal_paths = suboptimal_paths
		self.frame = frame
		self.node_spheres = node_spheres
		self.src_node_sphere = src_node_sphere
		self.sink_node_sphere = sink_node_sphere
		self.node_atom_representations = (
			node_atom_representations
		)
		self.radii = radii
		self.color = color 
		self.path_indices = path_indices


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

	def to_tcl(
			self,
			tcl: Optional[str] = '',
			new_line: Optional[bool] = True,
	) -> str:
		tcl = ''.join([tcl, self.__repr__()]) 
		if new_line:
			return tcl
		return tcl[:-1]	

class VisualizeCorrelationMatrix(serializer):

	def __init__(
			self,
			correlation_matrix: Union[str, np.ndarray],
			nodes: Nodes,
			reference_node_index: int,
			color: Optional[Tuple[str]] = (
				'red', 
				'yellow', 
				'green'
			),
			reference_node_representation: Optional[
				VmdRepresentation
			] = False,
			reference_node_sphere: Optional[
				vmdtcl.Sphere
			] = False,
			node_indices: Optional[List[int]] = False,
	) -> None:
		self.correlation_matrix = correlation_matrix
		self.nodes = nodes
		self.reference_node_index = reference_node_index
		self.color = color
		self.reference_node_representation = reference_node_representation
		self.reference_node_sphere = reference_node_sphere
		self.node_indices = node_indices

		
def get_src_and_sink_node_coordinates(
		path: Union[
			SuboptimalPaths, Path
		],
		frame: int,
) -> Tuple[np.ndarray]:
	if isinstance(path, SuboptimalPaths):
		src_coordinates = path[0][0][0].coordinates[
			frame
		]
		sink_coordinates = path[0][-1][1].coordinates[
			frame
		]
	else:
		src_coordinates = path[0][0].coordinates[
			frame
		]
		sink_coordinates = path[-1][1].coordinates[
			frame
		]
	return (
		src_coordinates,
		sink_coordinates
	)

def get_node_coordinates(
		path: Path,
		frame: int,
) -> Tuple[np.ndarray]:
	src_coordinates, sink_coordinates = (
		get_src_and_sink_node_coordinates(
			path,
			frame,
		)
	)
	coordinates = [src_coordinates]
	for edge in path.edges[1:]:
		coordinates.append(edge.node1.coordinates[frame])
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

def get_sorted_correlation_node_coordinates(
		correlation_matrix: np.ndarray,
		nodes: Nodes,
		node_index: int,
		frame: int,
) -> np.ndarray:
	node_correlation_dict = {
		i : correlation_matrix[node_index][i]
		for i in range(correlation_matrix.shape[0])
	}
	sorted_node_correlation_dict = {
		i : correlation for i, correlation 
		in sorted(
			node_correlation_dict.items(), 
			key=lambda item: item[1]
		)
	}
	sorted_correlation_node_coordinates = np.zeros(
		(correlation_matrix.shape[0], 3),
		dtype=np.float64
	)
	j = 0

	for i in list(sorted_node_correlation_dict):
		sorted_correlation_node_coordinates[j][0] = (
			nodes[i].coordinates[frame][0]
		)
		sorted_correlation_node_coordinates[j][1] = (
			nodes[i].coordinates[frame][1]
		)
		sorted_correlation_node_coordinates[j][2] = (
			nodes[i].coordinates[frame][2]
		)
		j += 1
	return sorted_correlation_node_coordinates

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

def visualize_correlation_matrix(
		parameters: VisualizeCorrelationMatrix,
		proc_name: Optional[str] = 'draw_correlation_matrix',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str: 
	correlation_matrix = parameters.correlation_matrix
	reference_node_index = parameters.reference_node_index
	nodes = parameters.nodes
	gradient_colors_matrix = np.array([
		Color(parameters.color[0]).rgb,
		Color(parameters.color[1]).rgb,
		Color(parameters.color[2]).rgb,
	], dtype=np.float64)
	c0x, c0y, c0z = gradient_colors_matrix[0]
	c1x, c1y, c1z = gradient_colors_matrix[1]
	c2x, c2y, c2z = gradient_colors_matrix[2]
	reference_node_representation = (
		parameters.reference_node_representation
	)
	reference_node_sphere = (
		parameters.reference_node_sphere
	)
	node_indices = parameters.node_indices

	if isinstance(correlation_matrix, str):
		correlation_matrix = np.load(correlation_matrix)
		
	tcl += f'proc {proc_name} {{}} {{\n'
	tcl += (
		f'proc lerpcolor {{ col1 col2 alpha }} {{\n'
		+ f'	set dc [vecsub $col2 $col1]\n'
		+ f'	set nc [vecadd $col1 [vecscale $dc $alpha]]\n'
		+ f'	return $nc'
		+ f'}}\n'
		+ f'proc coltogs {{ col }} {{\n'
		+ f'	foreach {{r g b}} $col {{}}\n'
		+ f'		set gray [expr ($r + $g + $b) / 3.0]\n'
		+ f'	return [list $gray $gray $gray]\n'
		+ f'}}\n'
		+ f'proc tricolor_scale {{}} {{\n'
		+ f'	display update off\n'
		+ f'	set mincolorid [expr [colorinfo num] - 1]\n'
		+ f'	set maxcolorid [expr [colorinfo max]]\n'
		+ f'	set colrange [expr $maxcolorid - $mincolorid]\n'
		+ f'	set colhalf [expr $colrange / 2]\n'
		+ f'	for {{set i $mincolorid}} {{$i < $maxcolorid}} '
		+ f'	{{incr i}} {{\n'
		+ f'		set colpcnt [expr ($i - $mincolorid) '
		+ f'/ double($colrange)]\n'
		+ f'		set firstColor {{{c0x} {c0y} {c0z}}}\n'
		+ f'		set middleColor {{{c1x} {c1y} {c1z}}}\n'
		+ f'		set lastColor {{{c2x} {c2y} {c2z}}}\n'
		+ f'		if {{ $colpcnt < 0.5 }} {{\n'
		+ f'			set nc [lerpcolor $firstColor $middleColor '
		+ f'[expr $colpcnt * 2.0]]\n'
		+ f'		}}'
		+ f'		else {{\n'
		+ f'			set nc [lerpcolor $middleColor $lastColor '
		+ f'[expr ($colpcnt-0.5) * 2.0]]\n'
		+ f'		}}\n'
		+ f'		foreach {{r g b}} $nc {{}}\n'
		+ f'		display update ui\n'
		+ f'		color change rgb $i $r $g $b\n'
		+ f'	}}\n'
		+ f'	display update on\n'
		+ f'}}\n'
	)
	if not node_indices:
		node_indices = [
			node.index for node in nodes.nodes
		]
	tcl += (
		f'set sel [atomselect top "all"]\n'
		+ f'$sel set beta 0.0\n'
		+ f'$sel delete\n'
	)
	for node in nodes:
		if node.index not in node_indices:
			continue
		tcl = vmdtcl.atomselect(
			f'sel', (
				"index", 
				node.atom_indices
			), tcl=tcl
		)
		correlation_value = correlation_matrix[
			reference_node_index,
			node.index
		]
		if correlation_value == np.inf:
			correlation_value = 99999999.999
		tcl += (
			f'$sel set beta '
			+ f'{correlation_value}\n'
		)
		tcl += f'$sel delete \n'
	tcl += f'tricolor_scale\n'
	if reference_node_representation:
		reference_node_representation.selection = (
			'index',
			nodes[node_index].atom_indices,
		)
		tcl = reference_node_representation.to_tcl(tcl=tcl)
	tcl += '}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def get_suboptimal_paths_nodes_atom_indices(
		suboptimal_paths: SuboptimalPaths,
) -> List[List[int]]:
	nodes_atom_indices = []
	for path in suboptimal_paths:
		nodes_atom_indices.append(
			path.edges[0].node1.atom_indices
		)
		for edge in path.edges:
			nodes_atom_indices.append(
				edge.node2.atom_indices
			)
	return nodes_atom_indices

def draw_suboptimal_paths(
		parameters: VisualizeSuboptimalPaths,
		proc_name: Optional[str] = 'draw_suboptimal_paths',
		cat_node_atom_representations: Optional[bool] = False,
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	frame = parameters.frame
	suboptimal_paths = parameters.suboptimal_paths
	frame_index_dict = {
		suboptimal_paths[0][0][0].coordinate_frames[
			frame_index
		] : frame_index for frame_index in
		range(len(
			suboptimal_paths[0][0][0].coordinate_frames
		))	
	}
	frame = frame_index_dict[frame]
	num_paths = len(suboptimal_paths)
	if parameters.path_indices:
		num_paths = len(parameters.path_indices)
	else:
		parameters.path_indices = [
			i for i in range(num_paths)
		]
	src_node_coordinates, sink_node_coordinates = (
		get_src_and_sink_node_coordinates(
			suboptimal_paths,
			frame,
		)
	)	
	tcl += f'proc {proc_name} {{}} {{\n'
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
	if parameters.node_atom_representations:
		nodes_atom_indices = (
			get_suboptimal_paths_nodes_atom_indices(
				parameters.suboptimal_paths
			) 
		)
		if cat_node_atom_representations:
			first_node_atom_rep_name = (
				list(parameters.node_atom_representations)[0]
			)
			cat_node_atom_representation = VmdRepresentation(
				**parameters.node_atom_representations[
					first_node_atom_rep_name
				].__dict__
			) 
			cat_node_atom_representation.selection[1].clear()
			for node_atom_representation in \
			parameters.node_atom_representations.values():
				if node_atom_representation.selection[1] \
				in nodes_atom_indices:
					for atom_index in \
					node_atom_representation.selection[1]:
						cat_node_atom_representation.selection[
							1
						].append(atom_index)
			tcl += cat_node_atom_representation.to_tcl()
		else:
			for node_atom_representation in \
			parameters.node_atom_representations.values():
				if node_atom_representation.selection[1] \
				in nodes_atom_indices:
					tcl += node_atom_representation.to_tcl()
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
	for path_index in parameters.path_indices:
		path = suboptimal_paths[path_index]
		node_coordinates = get_node_coordinates(path, frame)
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
	tcl += '}\n'
	if new_line:
		return tcl
	return tcl[:-1]

def visualize_nodes(
		suboptimal_paths: SuboptimalPaths,
		style: Optional[Union[
			vmdtcl.Lines,
			vmdtcl.Licorice,
			vmdtcl.VDW,
			vmdtcl.NewCartoon,
			vmdtcl.QuickSurf
		]] = vmdtcl.Lines(),
		coloring_method: Optional[str] = 'name',
		material: Optional[str] = 'Opaque',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True
) -> str:
	node_indices = set([]) 
	for path in suboptimal_paths:
		for edge in path:
			for node in edge:
				node_indices_size = len(node_indices)
				node_indices.add(node.index)
				if len(node_indices) != node_indices_size:	
					node_representation = VmdRepresentation(
						selection=("index", node.atom_indices), 
						style=style,
						coloring_method=coloring_method,
						material=material
					)
					tcl = node_representation.to_tcl(tcl=tcl)  
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
		tcl = initial_representation.to_tcl(tcl=tcl)  
	if new_line:
		return tcl
	return tcl[:-1]
	
	


