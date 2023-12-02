from model import *
import moderngl as mgl
import glm
import pygame as pg

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
		self.level = np.zeros((16,64,16),dtype='int8')
		self.level[:,:2,:]=1

		# n, s = 30, 3
		# for x in range(-n, n, s):
		# 	for z in range(-n, n, s):
		# 		add(Cube(app, pos=(x, -s, z)))
		for x, matrix in enumerate(self.level):
			for y, row in enumerate(matrix):
				for z, element in enumerate(row):
					if element!=0:
						add(Cube(app, pos=(x,y,z)))

		add(Cat(app, pos=(0, -2, -10)))
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

	def remove_block(self,block_coord):

		for i, cube in enumerate(self.objects):
			if isinstance(cube, Cube) and cube.pos == block_coord:
				self.objects.pop(i)
				break

		self.set_block_id(block_coord[0],block_coord[1],block_coord[2], 0)
	
	def add_block(self,block_coord,block_id):
		new_cube = Cube(self.app, pos=block_coord)
		self.objects.append(new_cube)
		self.set_block_id(block_coord[0],block_coord[1],block_coord[2], block_id)

	def get_block_id(self, x: int, y: int, z: int):
		min_x, min_y, min_z = (0,0,0)
		max_x,max_y, max_z = (16,64,16); 
		if min_x <= x < max_x and min_y <= y < max_y and min_z <= z < max_z:
			return self.level[x,y,z]
		else:
			return -1
	
	def set_block_id(self, x: int, y: int, z: int, block_id):
		min_x, min_y, min_z = (0,0,0)
		max_x,max_y, max_z = (16,64,16); 
		if min_x <= x < max_x and min_y <= y < max_y and min_z <= z < max_z and block_id>=0:
			self.level[x,y,z]=block_id
			return True
		else:
			return False

	def remove_first_block(self, block_coord_list):
		is_block = np.array([self.get_block_id(*c)>0 for c in block_coord_list])
		block_idxs = np.where(is_block)[0]
		if len(block_idxs)==0:
			return False
		else:
			idx0 = block_idxs[0]
			block_coord = block_coord_list[idx0]
			self.remove_block(block_coord)
			return True
	
	def add_first_block(self, block_coord_list):
		is_block = np.array([self.get_block_id(*c)>0 for c in block_coord_list])
		block_idxs = np.where(is_block)[0]
		if len(block_idxs)==0:
			return False
		elif block_idxs[0]==0: #!!Redo: Not a good rule for not placing a block. should only check if chord is within players bounding box
			return False
		else:
			idx0 = block_idxs[0]-1
			block_coord = block_coord_list[idx0]
			is_added = self.add_block(block_coord,1)
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
				print(grid_coordinates)
				print(current_position)
				print(direction)
				print(start_pos)

			current_position += direction * t_max[min_index]+eps_vec

		return intersected_coordinates

	



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

