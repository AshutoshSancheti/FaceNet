# All the functions have been tested for correct output
import dlib
import sys

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2
import numpy as np

import align_dlib


predictor_model = "shape_predictor_68_face_landmarks.dat"


class Detect_Face:
	def __init__(self, predictor, dims):
		#self.predictor_model = predictor
		self.face_detector = dlib.get_frontal_face_detector()
		self.face_pose_predictor  = dlib.shape_predictor(predictor)
		self.face_aligner = align_dlib.AlignDlib(predictor_model)
		self.output_dimensions = dims
		#self.win = dlib.image_window() #used to create an image window
		
	def detect_face(self, image):
		detected_faces = self.face_detector(image, 1)
		#print("I found {} faces in the file".format(len(detected_faces)))
		#self.win.set_image(image)
		
		for i, face_rect in enumerate(detected_faces):
			print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(),   face_rect.bottom()))
			pose_landmarks = self.face_pose_predictor(image, face_rect)
			alignedFace = self.face_aligner.align(self.output_dimensions, image, face_rect, landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)
			return alignedFace
			#self.win.add_overlay(face_rect)
			
		#dlib.hit_enter_to_continue()
		#print("Exiting the function")
		

	def multi_image_detect_face(self, file_names):
		"""
		file_names : list containing the paths of the images
		"""
		faces = []
		#print (len(file_names))
		for file_name in file_names:
			image = cv2.imread(file_name) #image size [250, 250, 3]
			#print(image.shape)
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			detected_faces = self.face_detector(image, 1)
			for i, face_rect in enumerate(detected_faces):
				pose_landmarks = self.face_pose_predictor(image, face_rect)
				alignedFace = self.face_aligner.align(self.output_dimensions, image, face_rect, landmarkIndices=align_dlib.AlignDlib.OUTER_EYES_AND_NOSE)
				faces.append(alignedFace.astype(np.float32))
				break
		face_stack = np.stack(faces, axis=0) # makes the size [3*BATCH_SIZE,299,299,3]
		return face_stack

if __name__ == '__main__':
	file_name = "lfw/Aaron_Peirsol/Aaron_Peirsol_0002.jpg"
	image = cv2.imread(file_name)
	#image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
	test = Detect_Face(predictor_model,160)
	alignedFace = test.detect_face(image)
	print (alignedFace.shape, alignedFace.dtype)
	cv2.imwrite('aligned_face.jpg',alignedFace)
	cv2.imshow('Aligned face', alignedFace)
	cv2.waitKey(0)
	cv2.destroyAllWindows
