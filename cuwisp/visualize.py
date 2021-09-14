from colour import Color
from typing import Optional, Tuple, Union, NamedTuple, Dict, List
from cuwisp.paths import SuboptimalPaths
from cuwisp.paths import Path
from cuwisp.nodes import Nodes 
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

class VisualizeCorrelationMatrix(serializer):

	def __init__(
			self,
			node_index: int,
			correlation_matrix_filename: str,
			nodes_xml_filename: str,
			color: Union[Tuple[str]],
			sphere_radius: Optional[float] = 1.0,
			sphere_resolution: Optional[int] = 25 ,
			node_color: Optional[Union[str, int]] = False,
			node_sphere_radius: Optional[float] = 1.0,
			num_nodes: Optional[int] = 1000,
			
	) -> None:
		self.nodes_xml_filename = nodes_xml_filename
		self.correlation_matrix_filename = (
			correlation_matrix_filename
		)
		self.color = color 
		self.sphere_radius = sphere_radius
		self.sphere_resolution = sphere_resolution 
		self.node_index = node_index
		self.node_color = node_color
		self.node_sphere_radius = node_sphere_radius
		self.num_nodes = num_nodes

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

def get_sorted_correlation_node_coordinates(
		correlation_matrix: np.ndarray,
		nodes: Nodes,
		node_index: int,
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
	for i in sorted_node_correlation_dict.keys():
		sorted_correlation_node_coordinates[i][0] = (
			nodes[i].coordinates[0]
		)
		sorted_correlation_node_coordinates[i][1] = (
			nodes[i].coordinates[1]
		)
		sorted_correlation_node_coordinates[i][2] = (
			nodes[i].coordinates[2]
		)
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
		molid: Optional[int] = 'top',
		tcl: Optional[str] = '',
		new_line: Optional[bool] = True,
) -> str:
	correlation_matrix = np.loadtxt(
		parameters.correlation_matrix_filename
	)
	nodes = Nodes()
	nodes.deserialize(
		parameters.nodes_xml_filename
	)
	num_nodes = parameters.num_nodes 
	sphere_resolution = parameters.sphere_resolution
	sphere_radius = parameters.sphere_radius
	node_index = parameters.node_index
	coordinates = get_sorted_correlation_node_coordinates(
		correlation_matrix,
		nodes,
		node_index,
	)	
	if len(coordinates) > num_nodes:
		coordinates = coordinates[:num_nodes]
	color_gradient = get_color_gradient(
		*parameters.color,
		len(coordinates)
	)
	tcl = vmdtcl.create_color_gradient(
		'color_gradient',
		color_gradient,
		tcl=tcl
	)
	color_index = 0
	for node_coordinates in coordinates:
		tcl += (
			f'set color_index [lindex '
			+ f'$color_gradient {color_index}]\n'
			+ f'graphics {molid} color $color_index]\n'
		)
		tcl = vmdtcl.Sphere(
			node_coordinates,
			radius=sphere_radius,
			resolution=sphere_resolution,
			
		).tcl(tcl=tcl)	
		color_index += 1
	if parameters.node_color:
		tcl = vmdtcl.set_draw_color(
			parameters.node_color, 
			tcl=tcl
		)
		tcl = vmdtcl.Sphere(
			nodes[node_index].coordinates,
			radius=parameters.node_sphere_radius,
			resolution=sphere_resolution,
			
		).tcl(tcl=tcl)	
	if new_line:
		return tcl
	return tcl[:-1]


def draw_suboptimal_paths(
		parameters: VisualizeSuboptimalPaths,
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
	
	


