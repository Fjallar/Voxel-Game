from model import *
import moderngl as mgl
import glm
import pygame as pg
from typing import Tuple, Iterable
from math import floor
from level import Level

class Scene:
	def __init__(self, app):
		self.app = app
		self.objects = []
		self.load()
		self.skybox = SkyBox(app)
	def add_object(self, obj):
		self.objects.append(obj)

	def load(self):
		app = self.app
		add = self.add_object
		self.level = np.zeros((16,64,16),dtype=np.uint8)
		self.LevelObj = Level(self.app)
		add(self.LevelObj)
		# self.level[:,:2,:]=1

		# add(Cat(app, pos=(0, -2, -10)))
		add(SkyBox(app))

	def render(self):
		for obj in self.objects:
			obj.render()
	
	def on_lclick(self):
		start_pos = self.app.camera.position
		direction = self.app.camera.forward
		length=4
		block_coord_list = self.raycast_get_block_coords(start_pos, direction, length)
		self.remove_first_block(block_coord_list)
	
	def on_rclick(self):
		start_pos = self.app.camera.position
		direction = self.app.camera.forward
		length=4
		block_coord_list = self.raycast_get_block_coords(start_pos, direction, length)
		self.add_first_block(block_coord_list)

	def remove_first_block(self, block_coord_list):
		is_block = np.array([self.LevelObj.get_block_id(coord)>0 for coord in block_coord_list])
		block_idxs = np.where(is_block)[0]
		if len(block_idxs)==0:
			return False
		else:
			idx0 = block_idxs[0]
			block_coord = block_coord_list[idx0]
			self.LevelObj.remove_block(block_coord)
			return True
	
	def add_first_block(self, block_coord_list):
		is_block = np.array([self.LevelObj.get_block_id(coord)>0 for coord in block_coord_list])
		block_idxs = np.where(is_block)[0]
		if len(block_idxs)==0:
			return False
		elif block_idxs[0]==0: #!!Redo: Not a good rule for not placing a block. should only check if chord is within players bounding box
			return False
		else:
			idx0 = block_idxs[0]-1
			block_coord = block_coord_list[idx0]
			is_added = self.LevelObj.add_block(block_coord,1)
			return is_added

	def raycast_get_block_coords(self, start_pos, direction, length):
		direction = np.array(direction)
		direction /= np.linalg.norm(direction)
		current_position = np.array(start_pos)
		eps_vec = np.sign(direction)*1e-6
		intersected_coordinates = []

		while np.linalg.norm(start_pos-current_position)<length:
			# -0.1 -> 0, which is wrong!, should be -1 --> FIXIT
			grid_coordinates = tuple(map(int,map(np.floor, current_position)))
			intersected_coordinates.append(grid_coordinates)

			# get next cube boundary with minimal distance (t_max) 
			t_max = np.empty(3)
			for i in range(3):
				if direction[i] > 0:
					t_max[i] = (grid_coordinates[i] + 1 - current_position[i]) / direction[i]
				elif direction[i] < 0:
					t_max[i] = (grid_coordinates[i] - current_position[i]) / direction[i]
				else:
					t_max[i] = float('inf')
			min_index = np.argmin(t_max)

			if t_max[min_index] <0:
				pg.event.set_grab(False)
				pg.mouse.set_visible(True)

			current_position += direction * t_max[min_index]+eps_vec

		return intersected_coordinates

# from perlin import perlin_fractal

# class Chunk:
# 	def __init__(self, coord, level):
# 		self.coord=(coord[0]*16,0,coord[1]*16)
# 		self.size=(16,64,16)
# 		self.chunk = self.gen_chunk()
# 		self.objects=[]
# 		self.gen_model(level.app)
	
# 	def gen_chunk(self):
# 		pos_grid = np.mgrid[self.coord[0]:self.coord[0]+self.size[0]+1,self.coord[2]:self.coord[2]+self.size[2]+1]
# 		h_map_unscaled =  perlin_fractal(pos_grid)
# 		h_map = (h_map_unscaled+1)*10+2

# 		# chunk = np.zeros((h_map.shape[0], self.size[1], h_map.shape[1]),dtype=np.uint8)
		
# 		chunk = np.zeros(self.size,dtype=np.uint8)
# 		for y in range(16):
# 			chunk[:, y, :] = (h_map > y)
# 		return chunk[:16,:,:16]
	
# 	def __contains__(self, in_coords: Iterable[int])->bool:
# 		assert len(in_coords)==3
# 		return all([self_x<=in_x<self_x+chunk_x for in_x, self_x, chunk_x in zip(in_coords, self.coord, self.size)])

# 	def convert_coord(self, coord: Tuple[int,int,int]):
# 		return tuple([x-self_x for x, self_x in zip(coord, self.coord)])

# 	def get_block_id(self, coord: Tuple[int, int, int])->int:
# 		if coord not in self:
# 			raise IndexError("Index out of bounds for this chunk. Current ")
# 		chunk_coord=self.convert_coord(coord)
# 		return self.chunk[*chunk_coord]

# 	def remove_block(self, coord: Tuple[int,int,int]):
# 		if coord not in self:
# 			raise IndexError("Index out of bounds for this chunk. Current ")
# 		chunk_coord=self.convert_coord(coord)
# 		self.chunk[*chunk_coord]=0
# 		for i, cube in enumerate(self.objects):
# 			if isinstance(cube, Cube) and cube.pos == coord:
# 				self.objects.pop(i)
# 				break

# 	def add_block(self, app, coord: Tuple[int, int, int], block_id: int):
# 		if coord not in self:
# 			raise IndexError("Index out of bounds for this chunk. Current ")
# 		chunk_coord=self.convert_coord(coord)
# 		self.chunk[*chunk_coord]=block_id
# 		self.objects.append(Cube(app, pos=coord))

	# def gen_model(self,app):
	# 	add = self.objects.append
	# 	for x, matrix in enumerate(self.chunk):
	# 		for y, row in enumerate(matrix):
	# 			for z, element in enumerate(row):
	# 				if element!=0:
	# 					add(Cube(app, pos=(x+self.coord[0],y+self.coord[1],z+self.coord[2])))
# 	def render(self):
# 		for object in self.objects:
# 			object.render()
		
# class Level:
# 	chunk_size=(16,16)

# 	def __init__(self, app):
# 		self.app=app
# 		self.ctx = app.ctx
# 		self.seed = 43
# 		self.chunks = []
# 		for i in range(2):
# 			for j in range(2):
# 				self.chunks.append(Chunk((i,j), self))
	
# 	def render(self):
# 		for chunk in self.chunks:
# 			chunk.render()
	
# 	def get_block_id(self, coord: Tuple[int, int, int])->int:
# 		for chunk in self.chunks:
# 			if coord in chunk:
# 				return chunk.get_block_id(coord)
# 		#should probably generate a new chunk and try once more.
# 		return -1
	
# 	#should be done by getters and setters.
# 	def remove_block(self,  coord: Tuple[int,int,int])->None:
# 		for chunk in self.chunks:
# 			if coord in chunk:
# 				chunk.remove_block(coord)
# 				return
	
# 	def add_block(self, coord: Tuple[int,int,int], block_id:int)->None:
# 		for chunk in self.chunks:
# 			if coord in chunk:
# 				chunk.add_block(self.app, coord, block_id)
# 				return
			
class ShadowMap:
	def __init__(self, app,lights):
		self.ctx = app.ctx
		self.app=app
		self.lights = lights

	def get_fbo(self,shadow_width=1024,shadow_height=1024):
		depth_texture = self.ctx.texture((shadow_width, shadow_height), components=1, dtype='f4')
		depth_texture.repeat_x = False
		depth_texture.repeat_y = False
		depth_texture.filter = (mgl.LINEAR, mgl.LINEAR)
		fbo = self.ctx.framebuffer(depth_attachments=depth_texture)
		fbo.bind()

if __name__ == '__main__':
	my_level = Level()
	print(my_level.chunks[0])