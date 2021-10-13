import os
from typing import (
    Optional,
    Tuple,
    Any,
    Hashable,
    Dict,
    List,
    Callable,
    Union,
    Container,
)
import warnings

from abserdes import Serializer as serializer
from .nodes import Nodes
import inspect

class IOException(Exception):

    def __init__(
            self,
            fname_attr,
            fname,
            supported_exts,
    ) -> None:
        self.fname_attr = fname_attr
        self.fname = fname
        self.supported_exts = supported_exts

    def __str__(
            self
    ):
        return (
                f'\n{self.fname_attr}={self.fname}\n'
                + f'{self.fname_attr} must be the name of a '.upper()
                + f'{self.supported_exts} '
                + f'file.'.upper()
        )

def value_in_contains_object(
        value: Container,
        obj: Union[Container, str],
        dtype: Optional[Any] = False,
) -> bool:
    if dtype:
        if not isinstance(value, dtype):
            return False
    if any(_ in value for _ in obj):
        return True
    return False

def single_dtype_in_container(
        dtype: Any,
        container: Container,
        container_dtype: Optional[Any] = False,
) -> bool:
    if container_dtype:
        if not isinstance(container, container_dtype):
            return False
    if sum(
            [
                isinstance(val, dtype) for val in container
            ]
    ) == len(container):
        return True
    return False

def get_function_arguments(
        func: Callable,
) -> List:
    func_args = inspect.getfullargspec(func).args
    if 'self' in func_args:
        func_args.remove('self')
    return func_args

def format_single_word_arg_name(
        arg_name: str,
        case: str,
) -> str:
    if case == 'upper':
        if len(arg_name) == 1:
            return arg_name.upper()
        return f'{arg_name[0].upper()}{arg_name[1:]}'
    else:
        if len(arg_name) == 1:
            return arg_name.lower()
        return f'{arg_name[0].lower()}{arg_name[1:]}'

def pascal_case(
        *strs: Tuple,
) -> str:
    strs = [
        f'{str_}' for str_ in strs
    ]
    if len(strs) == 1:
        return format_single_word_arg_name(
            arg_name=strs[0],
            case='upper',
        )
    else:
        formatted_name = f'{strs[0][0].upper()}{strs[0][1:]}'
    for str_ in strs[1:]:
        if len(str_) == 1:
            formatted_name += f'{str_.upper()}'
        else:
            formatted_name += (
                    f'{str_[0].upper()}'
                    + f'{str_[1:]}'
            )
    return formatted_name

def pascal_snake_case(
        *strs: Tuple,
) -> str:
    strs = [
        f'{str_}' for str_ in strs
    ]
    if len(strs) == 1:
        return format_single_word_arg_name(
            arg_name=strs[0],
            case='upper',
        )
    if len(strs[0]) == 1:
        formatted_name = f'{strs[0].upper()}_'
    else:
        formatted_name = f'{strs[0][0].upper()}{strs[0][1:]}_'
    for str_ in strs[1:]:
        if len(str_) == 1:
            formatted_name += f'{str_.upper()}_'
        else:
            formatted_name += (
                    f'{str_[0].upper()}'
                    + f'{str_[1:]}_'
            )
    return formatted_name[:-1]

def camel_case(
        *strs: Tuple,
) -> str:
    strs = [
        f'{str_}' for str_ in strs
    ]
    if len(strs) == 1:
        return format_single_word_arg_name(
            arg_name=strs[0],
            case='lower',
        )
    if len(strs[0]) == 1:
        formatted_name = f'{strs[0].lower()}'
    else:
        formatted_name = f'{strs[0][0].lower()}{strs[0][1:]}'
    for str_ in strs[1:]:
        if len(str_) == 1:
            formatted_name += f'{str_.upper()}'
        else:
            formatted_name += (
                    f'{str_[0].upper()}'
                    + f'{str_[1:]}'
            )
    return formatted_name

def camel_snake_case(
        *strs: Tuple,
) -> str:
    strs = [
        f'{str_}' for str_ in strs
    ]
    if len(strs) == 1:
        return format_single_word_arg_name(
            arg_name=strs[0],
            case='lower',
        )
    if len(strs[0]) == 1:
        formatted_name = f'{strs[0].lower()}_'
    else:
        formatted_name = f'{strs[0][0].lower()}{strs[0][1:]}_'
    for str_ in strs[1:]:
        if len(str_) == 1:
            formatted_name += f'{str_.upper()}_'
        else:
            formatted_name += (
                    f'{str_[0].upper()}'
                    + f'{str_[1:]}_'
            )
    return formatted_name[:-1]

def snake_case(
        *strs: Tuple,
) -> str:
    strs = [
        f'{str_}' for str_ in strs
    ]
    if len(strs) == 1:
        return format_single_word_arg_name(
            arg_name=strs[0],
            case='lower',
        )
    formatted_name = f''
    for str_ in strs:
        formatted_name += f'{str_}_'
    return formatted_name[:-1]

def format_directory_path(
        directory_path: str,
) -> str:
    if directory_path[-1] == '/':
        return directory_path[:-1]
    return directory_path

def naming_conventions() -> Dict:
    return {
        'pascal_case'      : pascal_case,
        'pascal_snake_case': pascal_snake_case,
        'camel_case'       : camel_case,
        'camel_snake_case' : camel_snake_case,
        'snake_case'       : snake_case,
    }

def set_fd_name(
        root_directory: str,
        calc_name: str,
        fd_base_name: Tuple[str],
        naming_convention: str,
        file_ext: Optional[str] = False
) -> str:
    naming_convention_args = (
        calc_name,
        *fd_base_name,
    )
    naming_convention = naming_conventions()[
        naming_convention
    ]
    fd_name = (
            f'{root_directory}/'
            + naming_convention(*naming_convention_args)
    )
    if file_ext:
        return f'{fd_name}{file_ext}'
    return fd_name

class FNC(serializer):

    def __init__(
            self,
            root_directory: str,
            calc_name: str,
            naming_convention: str,
    ) -> None:
        self.calc_name = calc_name
        self.naming_convention = naming_convention
        self.root_directory = root_directory
        if not os.path.exists(path=self.root_directory):
            os.makedirs(name=self.root_directory)
        self.splines_directory = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=('splines',),
            naming_convention=self.naming_convention,
        )
        self.graphics_directory = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=('graphics',),
            naming_convention=self.naming_convention,
        )
        self.suboptimal_paths_serialization_directory = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'suboptimal',
                'paths',
                'serialization',
            ),
            naming_convention=self.naming_convention,
        )
        self.node_coordinates_directory = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'node',
                'coordinates',
            ),
            naming_convention=self.naming_convention,
        )
        self.nodes_fname = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'nodes',
            ),
            naming_convention=self.naming_convention,
            file_ext='.xml',
        )
        self.correlation_matrix_fname = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'correlation',
                'matrix',
            ),
            naming_convention=self.naming_convention,
            file_ext='.npy',
        )
        self.contact_map_correlation_matrix_fname = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'contact',
                'map',
                'correlation',
                'matrix',
            ),
            naming_convention=self.naming_convention,
            file_ext='.npy',
        )
        self.contact_map_matrix_fname = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'contact',
                'map',
                'matrix',
            ),
            naming_convention=self.naming_convention,
            file_ext='.npy',
        )
        self.all_pairs_shortest_paths_matrix_fname = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'all',
                'pairs',
                'shortest',
                'paths',
                'matrix',
            ),
            naming_convention=self.naming_convention,
            file_ext='.npy',
        )
        self.suboptimal_paths_fname = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'suboptimal',
                'paths',
            ),
            naming_convention=self.naming_convention,
            file_ext='.xml',
        )
        self.io_fname = set_fd_name(
            root_directory=self.root_directory,
            calc_name=self.calc_name,
            fd_base_name=(
                'io',
            ),
            naming_convention=self.naming_convention,
            file_ext='.xml',
        )

    def __iter__(
            self
    ):
        for key, val in self.__dict__.items():
            yield key, val

    def __getitem__(
            self,
            key: Hashable,
    ) -> Any:
        return self.__dict__[key]

    def __setitem__(
            self,
            key: Hashable,
            val: Any,
    ) -> None:
        self.__dict__[key] = val

    def __contains__(
            self,
            key: Hashable
    ) -> bool:
        if key in self.__dict__.keys():
            return True
        return False

class IO(serializer):

    def __init__(
            self,
            calc_name: Optional[str] = False,
            root_directory: Optional[str] = False,
            trajectory_fname: Optional[str] = False,
            topology_fname: Optional[str] = False,
            naming_convention: Optional[str] = 'snake_case',
    ) -> None:
        if not calc_name:
            return
        self._fnc = FNC(
            calc_name=calc_name,
            naming_convention=naming_convention,
            root_directory=root_directory,
        )
        self.calc_name = self._fnc.calc_name
        self.root_directory = self._fnc.root_directory
        self.trajectory_fname = trajectory_fname
        self.topology_fname = topology_fname
        self.node_coordinates_directory = (
            self._fnc.node_coordinates_directory
        )
        self.graphics_directory = (
            self._fnc.graphics_directory
        )
        self.splines_directory = self._fnc.splines_directory
        self.suboptimal_paths_serialization_directory = (
            self._fnc.suboptimal_paths_serialization_directory
        )
        self.nodes_fname = self._fnc.nodes_fname
        self.suboptimal_paths_fname = self._fnc.suboptimal_paths_fname
        self.correlation_matrix_fname = (
            self._fnc.contact_map_matrix_fname
        )
        self.contact_map_correlation_matrix_fname = (
            self._fnc.contact_map_correlation_matrix_fname
        )
        self.contact_map_matrix_fname = (
            self._fnc.contact_map_matrix_fname
        )
        self.all_pairs_shortest_paths_matrix_fname = (
            self._fnc.all_pairs_shortest_paths_matrix_fname
        )
        self.io_fname = self._fnc.io_fname
        self.naming_convention = naming_convention
        self.file_extensions = {
            'trajectory_fname'                     : ['.pdb', '.dcd'],
            'topology_fname'                       : ['.pdb', '.psf'],
            'nodes_fname'                          : '.xml',
            'suboptimal_paths_fname'               : '.xml',
            'correlation_matrix_fname'             : '.npy',
            'contact_map_correlation_matrix_fname' : '.npy',
            'contact_map_matrix_fname'             : '.npy',
            'all_pairs_shortest_paths_matrix_fname': '.npy',
            'io_fname'                             : '.xml',
        }
        self.suboptimal_paths_fnames = False
        self.serialized_suboptimal_paths_directories = False
        self.serialized_correlation_matrices_directories = False
        self.serialize(
            xml_filename=self.io_fname,
        )

    def __iter__(
            self
    ):
        for key, val in self.__dict__.items():
            yield key, val

    def __getitem__(
            self,
            key: Hashable,
    ) -> Any:
        return self.__dict__[key]

    def __setitem__(
            self,
            key: Hashable,
            val: Any,
    ) -> None:
        self.__dict__[key] = val

    def __contains__(
            self,
            key: Hashable
    ) -> bool:
        if key in self.__dict__.keys():
            return True
        return False

    def __repr__(
            self
    ):
        fname_directory_attr_dict = {}
        attr_name_max_size = 0
        for attr_name, attr_val in self:
            if any(
                    str_ in attr_name for str_ in ('fname', 'dir')
            ):
                fname_directory_attr_dict[attr_name] = attr_val
                attr_name_max_size = max(
                    attr_name_max_size,
                    len(str(attr_name))
                )
        repr_str = f''
        kv_space = f''
        for _ in range(attr_name_max_size):
            kv_space += f' '
        for attr_name, attr_val in fname_directory_attr_dict.items():
            repr_str += (
                    f'{attr_name}'
                    + f'{kv_space[len(attr_name):]}'
                    + f':  {attr_val}\n'
            )
        return repr_str[:-1]

    def get_file_extension(
            self,
            fname_attr: str,
    ) -> str:
        base_fname = os.path.basename(
            getattr(self, fname_attr)
        )
        ext = base_fname.split('.').pop()
        if ext == base_fname:
            return ''
        return f'.{ext}'

    def get_supported_extensions(
            self,
            fname_attr: str,
    ) -> List:
        if not isinstance(
                self.file_extensions[fname_attr],
                list
        ):
            supported_exts = [
                self.file_extensions[fname_attr]
            ]
        else:
            supported_exts = (
                self.file_extensions[fname_attr]
            )
        return supported_exts

    def check_file_extension(
            self,
            fname: str,
            fname_attr,
            ext: str,
    ) -> bool:
        supported_exts = self.get_supported_extensions(
            fname_attr=fname_attr
        )
        if ext not in supported_exts:
            raise IOException(
                fname_attr=fname_attr,
                fname=fname,
                supported_exts=self.file_extensions[
                    fname_attr
                ]
            )
        return True

    def handle_missing_extension(
            self,
            fname: str,
            fname_attr,
    ) -> bool:
        if isinstance(
                self.file_extensions[fname_attr],
                list
        ):
            default_ext = self.file_extensions[
                fname_attr
            ][0]
        else:
            default_ext = self.file_extensions[fname_attr]
        warnings.warn(
            message=(
                    f'\n{fname_attr}={fname}\n'
                    + f'Extension for {fname_attr} '
                    + f'was not provided.\n'
                    + f'{fname_attr} is assumed to '
                    + f'be a {default_ext}' + f' file.\n'
                    + f'The updated value of {fname_attr} '
                      f'is\n'
                    + f'{fname}{default_ext}.'
            )
        )
        self[fname_attr] = f'{fname}{default_ext}'

    def check_file_extensions(
            self
    ) -> None:
        for fname_attr in self.file_extensions:
            fname = self[fname_attr]
            if fname:
                ext = self.get_file_extension(fname_attr)
                if ext != '':
                    self.check_file_extension(
                        fname=fname,
                        fname_attr=fname_attr,
                        ext=ext,
                    )
                else:
                    self.handle_missing_extension(
                        fname=fname,
                        fname_attr=fname_attr,
                    )

    def create_filetree(
            self,
            **kwargs: Dict,
    ) -> None:
        if kwargs:
            directories = []
            for directory in kwargs:
                if kwargs[directory]:
                    directories.append(self[directory])
        else:
            directories = [
                self['root_directory'],
                self['node_coordinates_directory'],
                self['suboptimal_paths_serialization_directory'],
                self['splines_directory'],
                self['graphics_directory'],
            ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def set_root_directory_update_list_attr(
            self,
            root_directory: str,
            attr_name: str,
            attr_val: List,
    ) -> None:
        for index in range(len(attr_val)):
            if self.root_directory in attr_val[index]:
                self[attr_name][index] = (
                    attr_val[index].replace(
                        self.root_directory,
                        root_directory
                    )
                )

    def set_root_directory_update_str_attr(
            self,
            root_directory: str,
            attr_name: str,
            attr_val: str,
    ) -> None:
        self[attr_name] = attr_val.replace(
            self.root_directory,
            root_directory,
        )
        if attr_name in self._fnc:
            self._fnc[attr_name] = self[attr_name]

    def set_root_directory_update_attr(
            self,
            root_directory: str,
            attr_name: str,
            attr_val: Union[str, List],
    ) -> None:
        if isinstance(attr_val, str):
            self.set_root_directory_update_str_attr(
                root_directory=root_directory,
                attr_name=attr_name,
                attr_val=attr_val,
            )
        else:
            self.set_root_directory_update_list_attr(
                root_directory=root_directory,
                attr_name=attr_name,
                attr_val=attr_val,
            )

    def update_root_directory(
            self,
            root_directory: str,
            attrs_to_update: Dict,
    ) -> None:
        root_directory = format_directory_path(
            root_directory
        )
        for attr_name, attr_val in attrs_to_update.items():
            self.set_root_directory_update_attr(
                root_directory=root_directory,
                attr_name=attr_name,
                attr_val=attr_val,
            )
        self.root_directory = root_directory
        self._fnc.root_directory = root_directory

    def set_updated_node_coordinates_directory(
            self,
            root_directory: str,
    ) -> bool:
        self.node_coordinates_directory = (
            self.node_coordinates_directory.replace(
                self.root_directory,
                root_directory,
            )
        )
        return True

    def update_nodes_xml(
            self,
            nodes_fname: str,
    ) -> bool:
        if os.path.exists(
                path=nodes_fname
        ):
            nodes = Nodes()
            nodes.deserialize(
                root=nodes_fname
            )
            for node in nodes:
                node.coordinates_directory = (
                    self.node_coordinates_directory
                )
            nodes.serialize(
                xml_filename=nodes_fname
            )
            return True
        return False

    def get_updated_nodes_fname(
            self,
            root_directory: str,
    ) -> str:
        nodes_fname = self.nodes_fname.replace(
            self.root_directory,
            root_directory
        )
        return nodes_fname

    def update_node_coordinates_directory(
            self,
            root_directory: str,
    ) -> bool:
        nodes_fname = self.get_updated_nodes_fname(
            root_directory=root_directory,
        )
        self.set_updated_node_coordinates_directory(
            root_directory=root_directory,
        )
        self.update_nodes_xml(
            nodes_fname=nodes_fname
        )
        return True

    def update_normal_attrs(
            self,
            normal_attrs: Dict,
    ) -> bool:
        for attr_name, attr_val in normal_attrs.items():
            if attr_val:
                self[attr_name] = attr_val
        return True

    def get_root_directory_update_attrs(
            self,
    ) -> Dict:
        attrs_to_update = {}
        attrs_to_ignore = [
            'root_directory',
            'trajectory_fname',
            'topology_fname',
        ]

        for attr_name, attr_val in self:
            if not value_in_contains_object(
                    value=attr_name,
                    obj=attrs_to_ignore,
            ):
                if isinstance(attr_val, str):
                    if value_in_contains_object(
                            value=self.root_directory,
                            obj=attr_val,
                            dtype=str,
                    ):
                        attrs_to_update[attr_name] = attr_val
                elif single_dtype_in_container(
                        dtype=str,
                        container=attr_val,
                        container_dtype=list
                ):
                    attrs_to_update[attr_name] = attr_val
        return attrs_to_update

    def update(
            self,
            root_directory: Optional[str] = False,
            trajectory_fname: Optional[str] = False,
            topology_fname: Optional[str] = False,
    ) -> bool:
        args = get_function_arguments(self.update)
        args.remove('root_directory')
        normal_attrs = {}
        for arg in args:
            normal_attrs[arg] = locals()[arg]
        self.update_normal_attrs(normal_attrs)
        if root_directory:
            self.update_node_coordinates_directory(
                root_directory=root_directory
            )
            attrs_to_update = self.get_root_directory_update_attrs()
            self.update_root_directory(
                root_directory=root_directory,
                attrs_to_update=attrs_to_update,
            )
        self.serialize(self.io_fname)
