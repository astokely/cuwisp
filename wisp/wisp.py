"""WISP is licensed under the Academic Free License 3.0. For more
information, please see http://opensource.org/licenses/AFL-3.0

WISP is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

Copyright 2012 Adam VanWart and Jacob D. Durrant. If you have any questions,
comments, or suggestions, please don't hesitate to contact durrantj [at] pitt
[dot] edu.

The latest version of WISP can be downloaded from
http://git.durrantlab.com/jdurrant/wisp

If you use WISP in your work, please cite A.T. Van Wart, J.D. Durrant, L.
Votapka, R.E. Amaro. Weighted implementation of suboptimal paths (WISP): An
optimized algorithm and tool for dynamical network analysis, J. Chem. Theory
Comput. 10 (2014) 511-517."""

'''
import copy
import shutil
import pstats
from pstats import SortKey
'''
import shutil
import mdtraj as md
import os
import sys
import numpy as np
from typing import Optional
from .correlation_matrix import GetCorrelationMatrix
from .paths import get_suboptimal_paths


def calculate_correlation_matrix(
		output_directory: str,
		contact_map_distance_limit: float,
		pdb_trajectory_filename: str,
		correlation_matrix_filename: Optional[str] = '',
		correlation_matrix_after_contact_map_filename: Optional[str] = '',
		nodes_xml_filename: Optional[str] = '',
) -> None:
	tmp_path = os.path.dirname(os.path.abspath(
		sys.modules[GetCorrelationMatrix.__module__].__file__)
	) + "/tmp"
	if os.path.exists(output_directory):
		shutil.rmtree(output_directory)
	os.makedirs(output_directory)

	correlation_matrix = GetCorrelationMatrix(
		output_directory,
		contact_map_distance_limit,
		pdb_trajectory_filename,
		tmp_path,	
		correlation_matrix_filename,
		correlation_matrix_after_contact_map_filename,
		nodes_xml_filename
	)
	shutil.rmtree(tmp_path)

def calculate_suboptimal_paths(
		input_files_path: str,
		src: int,
		sink: int,
		cutoff: Optional[float] = None,
		k: Optional[int] = 16,
		use_contact_map_correlation_matrix: Optional[bool] = True,
		correlation_matrix_filename: Optional[str] = '',
		nodes_xml_filename: Optional[str] = '',
		suboptimal_paths_xml_filename: Optional[str] = '',
) -> None:
		if correlation_matrix_filename == '':
			if use_contact_map_correlation_matrix: 
				correlation_matrix_file = (
					"correlation_matrix_after_contact_map.txt" 
				)
			else:
				correlation_matrix_file = "correlation_matrix.txt" 
		if nodes_xml_filename == '':
			nodes_xml_file = "nodes.xml"
		if suboptimal_paths_xml_filename == '':
			suboptimal_paths_xml_filename = "suboptimal_paths.xml" 
		get_suboptimal_paths(
			input_files_path, 
			correlation_matrix_file,
			nodes_xml_file,
			src, 
			sink,
			suboptimal_paths_xml_filename, 
			cutoff,
			k 
		)



