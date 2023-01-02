import cv2, time
import numpy as np 
import threading


class ImageFilterThread():
	def __init__(self, list_img_pipeline):
		super(ImageFilterThread, self).__init__()
		self.tweak_thread = True
		self.list_img_pipeline = list_img_pipeline
		self.list_end_img = []

	def start_thread(self):
		list_img_filter = []
		count = 0
		for i in self.list_img_pipeline:
			list_img_filter.append(IMGFilter(i[0], i[1]))
			threading.Thread(target=list_img_filter[count].start).start()
			count += 1
		while True:
			time.sleep(0.01)
			list_end_tweak = []
			for i in list_img_filter:
				list_end_tweak.append(i.work)
			if True not in list_img_filter:
				break

		self.list_end_img = list_img_filter

class IMGFilter():

	def __init__(self, img, pipeline_dict):
		super(IMGFilter, self).__init__()
		self.work = True
		self.pipeline_dict = pipeline_dict
		self.img = img
		self.end_img = None

	def start(self):
		self.pipeline()

	def init_generator(func):
		def generator_wrap(*args, **kwargs):
			generator = func(*args, **kwargs)
			next(generator)
			return generator
		return generator_wrap

	def pipeline(self):
		img_cor = self.append_img()
		if 'inversion' in self.pipeline_dict:
			self.inversion = self.pipeline_dict['inversion']
			img_cor = self.color_inversion(img_cor)
		if 'threshold' in self.pipeline_dict:
			self.threshold_list = self.pipeline_dict['threshold']
			img_cor = self.threshold_func(img_cor)
		if 'brightness' in self.pipeline_dict:
			self.brightness = self.pipeline_dict['brightness']
			img_cor = self.brightness_adjustment(img_cor)
		if 'resize' in self.pipeline_dict:
			self.resize = self.pipeline_dict['resize']
			img_cor = self.resize_img(img_cor)
		self.get_img(img_cor)

	def get_img(self, next_cor):
		next_cor.send(self.img)
		next_cor.close()

	@init_generator	
	def resize_img(self, next_cor):
		try:
			while True:
				img = yield
				next_cor.send(cv2.resize(img, self.resize, interpolation = cv2.INTER_LINEAR))
		except GeneratorExit:
			next_cor.close()

	@init_generator
	def brightness_adjustment(self, next_cor):
		try:
			while True:
				img = yield
				hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
				h, s, v = cv2.split(hsv)
				lim = 255 - self.brightness
				v[v > lim] = 255
				v[v <= lim] += self.brightness
				final_hsv = cv2.merge((h, s, v))
				next_cor.send(cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR))
		except GeneratorExit:
			next_cor.close()

	@init_generator
	def threshold_func(self, next_cor):
		try:
			while True:
				t_l = self.threshold_list
				img = yield
				ret, img = cv2.threshold(img, t_l[0], t_l[1], t_l[2])
				next_cor.send(img)
		except GeneratorExit:
			next_cor.close()

	@init_generator
	def color_inversion(self, next_cor):
		try:
			while True:
				img = yield
				next_cor.send(cv2.bitwise_not(img))
		except GeneratorExit:
			next_cor.close()	

	@init_generator
	def append_img(self):
		try:
			while True:
				img = yield
				self.end_img = img
		except GeneratorExit:
			#print('File processing is complete')
			self.work = False


if __name__ == "__main__":
	img = cv2.imread("data.jpg")
	img_filter = IMGFilter(img, {'resize':[600,283], 'brightness': 50, 'threshold':[127, 255, 0], 'inversion': True})
	img_filter.start()
	cv2.imwrite(f'./end_img/img.jpg',img_filter.end_img)

	list_img = []
	for i in range(1000):
		list_img.append([img, {'resize':[600,283], 'brightness': 50, 'threshold':[127, 255, 0], 'inversion': True}])

	print('Performance testing (dataset 1000 images)')
	start_time = time.time()
	for i in list_img:
		img_filter = IMGFilter(i[0], i[1])
		img_filter.start()
	end_time = time.time()
	print(f'Without multithreading {round(end_time-start_time, 3)}s')
	start_time = time.time()
	th_img = ImageFilterThread(list_img)
	th_img.start_thread()
	end_time = time.time()
	print(f'With multithreading {round(end_time-start_time, 3)}s')
	
	
	count = 0
	for i in th_img.list_end_img:
		cv2.imwrite(f'./end_img/img{count}.jpg', i.end_img)
		count += 1


