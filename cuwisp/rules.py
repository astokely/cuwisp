from collections import deque
import inspect
from typing import Any, Tuple, Optional, \
	List, Union, Set, Deque, Callable 

class Rule(object):

	def __init__(
			self, 
			append: Callable,
			pop: Callable, 
	) -> None:
		self._pop = pop
		self._append = append 

	def __repr__(self):
		return (
			f'{inspect.getsource(self._append)}\n'
			+ f'{inspect.getsource(self._pop)}'
		)

	def pop(
			self,
			dq: Deque,
	) -> Any:
		return self._pop(dq)

	def append(
			self,
			dq: Deque,
			val
	) -> None:
		return self._append(dq, val)
	
		




