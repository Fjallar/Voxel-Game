'''
Todos:
1. Have everything as an updated array
2. Generate and regenerate geometry with a 3D-convolution,


Details:
For generating mesh from inital beginnings:
kernel=
[
[[0,0,0 ],
[0,32,0],
[0,0,0,]],

[[0, 2,  0],
[8,64, 1],
[0, 16, 0]],

[[0,0,0],
[0,4,0],
[0,0,0]]]

Generate mesh from block being removed
kernel_inv (invert the kernel above and place the values in ...)



Beyond that. Updating should probably be done on gpu. Which seems difficult with OpenGL, and seems easier on CUDA or Vulkan, and perhaps
The things that needs to be saved are (for the moment) all of the blocks that have been added/removed




'''
from model import *
from typing import Tuple, Iterable
from perlin import perlin_fractal
import glm
import numpy as np
from shader_program import ShaderProgram
from numba import njit
# from scipy import ndimage





@njit
def is_occupied(pos: Tuple[int,int,int], level_array):
	level_max_idx = level_array.shape
	for dim, idx in enumerate(pos):
		if not (0<= idx < level_max_idx[dim]):
			return False
	i,j,k = pos
	return level_array[i,j,k]!=0


@njit
def get_pos_from_index(start_pos: Tuple[int,int,int], index3D: Tuple[int,int,int]) -> Tuple[int,int,int]:
	return (index3D[0]+start_pos[0], index3D[1]+start_pos[1], index3D[2]+start_pos[2])


# Define the chunk size and initial number of chunks
chunk_size = 16*16  # number of tuples per chunk
num_floats_per_tuple = 8


@njit
def append_face(vertex_array, data_face, num_current_tuples: int):
    # num_current_tuples = vertex_array.shape[0]
    num_new_tuples = data_face.shape[0]

    # Check if the existing array can accommodate the new data
    if num_current_tuples + num_new_tuples > vertex_array.shape[0]:
        # Calculate new required size
        new_size =2*vertex_array.shape[0]
        # Create a new array with the new size
        new_vertex_array = np.zeros((new_size, num_floats_per_tuple), dtype=np.float32)
        new_vertex_array[:num_current_tuples] = vertex_array[:num_current_tuples]
        vertex_array = new_vertex_array
    
    # Append new data
    vertex_array[num_current_tuples:num_current_tuples + num_new_tuples] = data_face
    return vertex_array, num_current_tuples + num_new_tuples

@njit
def generate_vertex_array(start_pos, level_array):
	# is_occupied = lambda i,j,k: self.level_array[i,j,k]!=0 if all([0<=idx<level_max_idx for idx, level_max_idx in zip((i,j,k), self.level_array.shape)]) else False
	vertex_array = np.zeros((chunk_size, num_floats_per_tuple), dtype='f4')
	# spos=start_pos
	num_current_tuples:int=0
	for x, matrix in enumerate(level_array):
		for y, row in enumerate(matrix):
			for z, element in enumerate(row):
				if element!=0:
					spos = get_pos_from_index(start_pos, (x,y,z))
					if not is_occupied((x-1,y,z),level_array):
						# (uv(2),  direction(3), vertex_pos)
						face_array=np.array([(0,0, -1,0,0, spos[0],spos[1]+1,spos[2]), (1,0, -1,0,0, spos[0],spos[1],spos[2]), (1,1, -1,0,0, spos[0],spos[1],spos[2]+1),
						   (0,0, -1,0,0, spos[0],spos[1]+1,spos[2]), (1,1, -1,0,0, spos[0],spos[1],spos[2]+1), (0,1, -1,0,0, spos[0],spos[1]+1,spos[2]+1)], dtype='f4')
						vertex_array, num_current_tuples = append_face(vertex_array,face_array, num_current_tuples)
					if not is_occupied((x+1,y,z),level_array):
						face_array= np.array([(0,0, 1,0,0, spos[0]+1,spos[1],spos[2]), (1,0, 1,0,0, spos[0]+1,spos[1]+1,spos[2]), (1,1, 1,0,0, spos[0]+1,spos[1]+1,spos[2]+1),
					 				   (0,0, 1,0,0, spos[0]+1,spos[1],spos[2]), (1,1, 1,0,0, spos[0]+1,spos[1]+1,spos[2]+1), (0,1, 1,0,0, spos[0]+1,spos[1],spos[2]+1)], dtype='f4') 
						vertex_array, num_current_tuples = append_face(vertex_array,face_array, num_current_tuples)

					if not is_occupied((x,y-1,z),level_array):
						face_array= np.array([(0,0, 0,-1,0, spos[0],spos[1],spos[2]), (1,0, 0,-1,0, spos[0]+1,spos[1],spos[2]), (1,1, 0,-1,0, spos[0]+1,spos[1],spos[2]+1),
					 				   (0,0, 0,-1,0, spos[0],spos[1],spos[2]), (1,1, 0,-1,0, spos[0]+1,spos[1],spos[2]+1), (0,1, 0,-1,0, spos[0],spos[1],spos[2]+1)], dtype='f4') 
						vertex_array, num_current_tuples = append_face(vertex_array,face_array, num_current_tuples)
					if not is_occupied((x,y+1,z),level_array):
						face_array= np.array([(0,0, 0,1,0, spos[0]+1,spos[1]+1,spos[2]), (1,0, 0,1,0, spos[0],spos[1]+1,spos[2]), (1,1, 0,1,0, spos[0],spos[1]+1,spos[2]+1),
					 				   (0,0, 0,1,0, spos[0]+1,spos[1]+1,spos[2]), (1,1, 0,1,0, spos[0],spos[1]+1,spos[2]+1), (0,1, 0,1,0, spos[0]+1,spos[1]+1,spos[2]+1)], dtype='f4') 
						vertex_array, num_current_tuples = append_face(vertex_array,face_array, num_current_tuples)

					if not is_occupied((x,y,z-1),level_array):
						face_array= np.array([(0,0, 0,0,-1, spos[0],spos[1]+1,spos[2]), (1,0, 0,0,-1, spos[0]+1,spos[1]+1,spos[2]), (1,1, 0,0,-1, spos[0]+1,spos[1],spos[2]),
					 				   (0,0, 0,0,-1, spos[0],spos[1]+1,spos[2]), (1,1, 0,0,-1, spos[0]+1,spos[1],spos[2]), (0,1, 0,0,-1, spos[0],spos[1],spos[2])], dtype='f4') 
						vertex_array, num_current_tuples = append_face(vertex_array,face_array, num_current_tuples)
					if not is_occupied((x,y,z+1),level_array):
						face_array= np.array([(0,0, 0,0,1, spos[0],spos[1],spos[2]+1), (1,0, 0,0,1, spos[0]+1,spos[1],spos[2]+1), (1,1, 0,0,1, spos[0]+1,spos[1]+1,spos[2]+1),
					 				   (0,0, 0,0,1, spos[0],spos[1],spos[2]+1), (1,1, 0,0,1, spos[0]+1,spos[1]+1,spos[2]+1), (0,1, 0,0,1, spos[0],spos[1]+1,spos[2]+1)], dtype='f4') 
						vertex_array, num_current_tuples = append_face(vertex_array,face_array, num_current_tuples)
	vertex_array = vertex_array[:num_current_tuples,:]
	return vertex_array




class Level:
	chunk_size: Tuple[int,int,int] = (16,64,16)
	nr_chunks: Tuple[int,int] = (9,9)
	size: Tuple[int,int,int] = (nr_chunks[0]*chunk_size[0], chunk_size[1], nr_chunks[1]*chunk_size[2]) 


	def __init__(self, app):
		self.app=app
		self.ctx=app.ctx
		self.chunk_index = self.get_camera_chunk_index()
		self.level_array=self.gen_from_perlin(self.chunk_from(), self.chunk_to())
		self.coord=(self.chunk_index[0]*self.chunk_size[0],0,self.chunk_index[1]*self.chunk_size[2])
		# self.objects=[]
		# self.gen_model(app)
		self.init_model()

	def update_model(self):
		# Regenerate vertex data
		new_vertex_array = generate_vertex_array(self.coord, self.level_array)
		
		# Check if the new data fits into the existing buffer
		if new_vertex_array.nbytes > self.vbo.size:
			self.vbo.orphan(size=new_vertex_array.nbytes)  # Reallocate buffer with new size

		self.vbo.write(new_vertex_array)
		

	def init_model(self):
		vertex_array = generate_vertex_array(self.coord, self.level_array)
		
		self.vbo = self.ctx.buffer(vertex_array)
		self.format = '2f 3f 3f'
		self.attribs = ['in_texcoord_0', 'in_normal', 'in_position']
		self.program = ShaderProgram(self.ctx).programs['default']
		self.vao = self.ctx.vertex_array(self.program, [(self.vbo, self.format, *self.attribs)])

		self.tex_id = 0
		self.texture = self.app.mesh.texture.textures[self.tex_id]
		self.program['u_texture_0']
		self.texture.use()
		self.m_model = glm.mat4()

		self.program['m_proj'].write(self.app.camera.m_proj)
		self.program['m_view'].write(self.app.camera.m_view)
		self.program['m_model'].write(self.m_model)
        # light
		self.program['light.position'].write(self.app.light.position)
		self.program['light.Ia'].write(self.app.light.Ia)
		self.program['light.Id'].write(self.app.light.Id)
		self.program['light.Is'].write(self.app.light.Is)


	def get_camera_chunk_index(self):
		chunk_pos = self.app.camera.position//16
		return (chunk_pos.x-self.nr_chunks[0]//2, chunk_pos.z-self.nr_chunks[1]//2)

	def chunk_from(self):
		return (self.chunk_index[0], self.chunk_index[1])

	def chunk_to(self):
		return (self.nr_chunks[0]+self.chunk_index[0], self.nr_chunks[0]+self.chunk_index[1])	

	def update(self):
		self.texture.use()
		self.program['camPos'].write(self.app.camera.position)
		self.program['m_view'].write(self.app.camera.m_view)
		self.program['m_model'].write(self.m_model)

		new_chunk_idx = self.get_camera_chunk_index()
		if self.chunk_index!=new_chunk_idx:
			self.chunk_index=new_chunk_idx
			self.coord=(self.chunk_index[0]*self.chunk_size[0],0,self.chunk_index[1]*self.chunk_size[2])
			self.level_array=self.gen_from_perlin(self.chunk_from(), self.chunk_to(),in_array=self.level_array)
			self.update_model()

	def generate_geometry():
		kernel = np.array([
			[[0,0,0],  [0,32,0],  [0,0,0,]],
			[[0,2,0],  [8,64,1],  [0,16,0]],
			[[0,0,0],   [0,4,0],   [0,0,0]]])
	
	def __contains__(self, in_coords: Iterable[int])->bool:
		assert len(in_coords)==3
		return all([self_x<=in_x<self_x+chunk_x for in_x, self_x, chunk_x in zip(in_coords, self.coord, self.size)])
	
	def convert_coord(self, coord: Tuple[int,int,int]):
		return tuple([int(x-self_x) for x, self_x in zip(coord, self.coord)])
	
	def get_pos(self, index3D: Tuple[int,int,int]):
		return tuple([i+self_x for i, self_x in zip(index3D, self.coord)])
		
	def get_block_id(self, coord: Tuple[int, int, int])->int:
		if coord not in self:
			raise IndexError("Index out of bounds for this chunk.")
		level_coord=self.convert_coord(coord)
		return self.level_array[*level_coord]
	
	def remove_block(self, coord: Tuple[int,int,int]):
		if coord not in self:
			raise IndexError("Index out of bounds for this chunk. Current ")
		level_coord=self.convert_coord(coord)
		self.level_array[*level_coord]=0
		self.update_model()
		
		# for i, cube in enumerate(self.objects):
		# 	if isinstance(cube, Cube) and cube.pos == coord:
		# 		self.objects.pop(i)
		# 		break
	
	def add_block(self, coord: Tuple[int, int, int], block_id: int):
		if coord not in self:
			raise IndexError("Index out of bounds for this chunk. Current ")
		level_coord=self.convert_coord(coord)
		self.level_array[*level_coord]=block_id
		# self.objects.append(Cube(app, pos=coord))
		self.update_model()
	
	def render(self):
		# for object in self.objects:
		# 	object.render()
		self.update()
		self.vao.render()

	def gen_from_perlin(self, chunk_startpos: Tuple[int,int], chunk_endpos: Tuple[int,int], in_array=None):
		if in_array is None:
			nr_chunks_x= chunk_endpos[0]-chunk_startpos[0]
			nr_chunks_y= chunk_endpos[1]-chunk_startpos[1]
			
			size: Tuple[int,int,int] = (int(nr_chunks_x * self.chunk_size[0]), int(self.chunk_size[1]), int(nr_chunks_y * self.chunk_size[2]))
			in_array=np.zeros(size,dtype=np.uint8)

		start_x = chunk_startpos[0]*self.chunk_size[0]; end_x=chunk_endpos[0]*self.chunk_size[0]
		start_y = chunk_startpos[1]*self.chunk_size[2]; end_y=chunk_endpos[1]*self.chunk_size[2]
		
		pos_grid = np.mgrid[start_x:end_x+1,start_y:end_y+1]
		h_map_unscaled =  perlin_fractal(pos_grid)
		h_map = (h_map_unscaled+1)*10+2
		for y in range(64):
			in_array[:, y, :] = (h_map > y)
		return in_array

