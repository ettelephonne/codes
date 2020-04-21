import pyrealsense2 as rs
import numpy as np
import cv2
import argparse

# converti les fichiers .bag de realsense vers des fichiers .avi. Attention, les données de profondeur sont perdues à jamais en .avi!!!!

parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
parser.add_argument("-i", "--input", type=str, default='/home/ettelephonne/Documents/20200419_153126.bag', help="Path to the bag file")
args = parser.parse_args()

# initialisation de l'écriture en .avi (changer adresse et noms)
outVid = cv2.VideoWriter('videos/auto_pietons_2_color.avi', cv2.VideoWriter_fourcc(*'XVID'),12,(640, 480))
outVid2 = cv2.VideoWriter('videos/auto_pietons_2_depth.avi', cv2.VideoWriter_fourcc(*'XVID'),12,(640, 480))

# initialisation de la lecture des fichiers .bag
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, args.input)

# il est impératif de connaître les formats des vidéos (taille, format et fps) sinon ça ne s'initialise pas
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)

# outils pour faire de belles couleurs
colorizer = rs.colorizer()

# il faut arrêter la boucle manuellement avec "q" :-(
while True:
	# checke qu'une frame est disponible
	frames = pipeline.wait_for_frames()

	# frames est un objet composite qui contient deux images (color et depth) dont on extrait les informations
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()

	# Ccolorisation
	depth_color_frame = colorizer.colorize(depth_frame)

	# conversion numpy, après c'est facile à manipuler
	depth_image = np.asanyarray(depth_color_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())

	# attention entre bgr et rgb
	color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

	# on affiche
	cv2.imshow("Profondeur", depth_image)
	cv2.imshow("Couleur", color_image)

	# on écrit
	outVid.write(color_image)
	outVid2.write(depth_image)

	key = cv2.waitKey(1)
	# on sort
	if key & 0xFF == ord('q') or key == 27:
		cv2.destroyAllWindows()
		break
outVid.release()
outVid2.release()
