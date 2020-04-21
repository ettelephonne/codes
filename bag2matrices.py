import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path
import glob

# pipeline.wait_for_frames attend qu'une frame soit disponible, PAS QUE LE CALCUL SOIT TERMINÉ POUR ALLER CHERCHER LA FRAME D'APRÈS
# Dans un fichier .bag, la frame d'après est toujours disponible... Les frames reçues pendant le temps de calcul SONT PERDUES comme s'il s'agissait d'un live stream
# "Queue" oblige à stocker les frames (attention mémoire) mais ne semble pas fonctionner pas avec les frames composites
 # solution: écrire les frames en matrices. Elles seront correctement lues les unes après les autres dans un autre code

parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
parser.add_argument("-i", "--input", type=str, default='/home/ettelephonne/Documents/20200420_114742.bag', help="Path to the bag file")
args = parser.parse_args()

# mettre un chiffre différent entre les différentes utilisations pour ne pas que les fichiers s écrasent.
n = input('numero du film à faire? = ')

# démarrage de la lecture du fichier .bag
pipeline = rs.pipeline()
config = rs.config()
rs.config.enable_device_from_file(config, args.input)

# attention, si tu connais pas la config au départ, rien ne va fonctionner
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)

# crée le fichier où les matrices seront stockées s'il n'existe pas
if not os.path.exists('/home/ettelephonne/projet_visionmeteo/codes/deep_depth/film_'+str(n)):
    os.makedirs('/home/ettelephonne/projet_visionmeteo/codes/deep_depth/film_'+str(n))

# détruit les fichiers à l'intérieur à chaque nouveau film
files = glob.glob('/home/ettelephonne/projet_visionmeteo/codes/deep_depth/film_'+str(n)+'/*')
for f in files:
    os.remove(f)

# attention il faut arrêter le vidéo manuellement, on ne connaît pas le nombre de frames
i = 0
while True:
	i = i+1
	# checke qu'une frame est disponible
	frames = pipeline.wait_for_frames()

	# frames est un objet composite qui contient deux images (color et depth) dont on extrait les informations
	depth_frame = frames.get_depth_frame()
	color_frame = frames.get_color_frame()

	# transformatin numpy pour une gestion facile
	depth_image = np.asanyarray(depth_frame.get_data())
	color_image = np.asanyarray(color_frame.get_data())

	# attention rgb et bgr
	color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

	# écriture des matrices
	np.save('/home/ettelephonne/projet_visionmeteo/codes/deep_depth/film_'+str(n)+'/film_'+str(n)+'_d_'+str(i)+'.txt',depth_image)
	np.save('/home/ettelephonne/projet_visionmeteo/codes/deep_depth/film_'+str(n)+'/film_'+str(n)+'_c_' + str(i) + '.txt', color_image)

	# affichage du film couleur
	cv2.imshow("Couleur", color_image)
	key = cv2.waitKey(1)

	# on peut quitter avec "q"
	if key & 0xFF == ord('q') or key == 27:
		cv2.destroyAllWindows()
		break

