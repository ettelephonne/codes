import numpy as np
import argparse
import cv2
import os


# mettre un chiffre différent entre les différentes utilisations pour ne pas que les fichiers s écrasent.
n = input('numero du film à faire? = ')

# Les options à remplir. Il faut changer les adresses
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=1,
	help="whether or not output frame should be displayed")
ap.add_argument("-o", "--output", default='/home/ettelephonne/projet_visionmeteo/codes/deep_depth/avi/film_'+str(n)+'.avi',
	help="base path to output directory")
ap.add_argument("-m", "--mask-rcnn", default="/home/ettelephonne/pyImage_search/opencv-dnn-gpu-examples/opencv-mask-rcnn-cuda/mask-rcnn-coco",
	help="base path to mask-rcnn directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.8,
	help="minimum threshold for pixel-wise mask segmentation")
ap.add_argument("-u", "--use-gpu", type=bool, default=1,
	help="boolean indicating if CUDA GPU should be used")
args = vars(ap.parse_args())

# charge les labels du st d entrainement coco
labelsPath = os.path.sep.join([args["mask_rcnn"],
	"object_detection_classes_coco.txt"])
LABELS = open(labelsPath).read().strip().split("\n")

# fait de belles couleurs pour chacun des labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# charge l architecture et les poids
weightsPath = os.path.sep.join([args["mask_rcnn"],
	"frozen_inference_graph.pb"])
configPath = os.path.sep.join([args["mask_rcnn"],
	"mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"])

# charge le réseau avec cv2
print("[INFO] loading Mask R-CNN from disk...")
net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

# met en route le calcul sur GPU (cv2 doit être buildé from source avec l'option CUDA = ON)
if args["use_gpu"]:
	print("[INFO] setting preferable backend and target to CUDA...")
	net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

writer = None
fnt = cv2.FONT_HERSHEY_PLAIN

# Donne le nombre de frames dans le vidéo * 2 (color et depth)
matrices = os.listdir('/home/ettelephonne/projet_visionmeteo/codes/deep_depth/film_'+str(n))

# calculs sur le nombre de frames
i = 0
for i in range(int(len(matrices)/2)):
	i +=1
	# pour faire du suivi, surtout que ça peut être long
	print('frame '+str(i)+' sur '+str(len(matrices)/2))

	# chargement des matrices de color et depth
	color_image = np.load('/home/ettelephonne/projet_visionmeteo/codes/deep_depth/film_'+str(n)+'/film_'+str(n)+'_c_'+str(i)+'.txt.npy')
	depth_image = np.load('/home/ettelephonne/projet_visionmeteo/codes/deep_depth/film_'+ str(n)+'/film_'+str(n)+'_d_'+ str(i) + '.txt.npy')

	# color va dans le blob (étape préparatoire pour le réseau de neurones)
	blob = cv2.dnn.blobFromImage(color_image, swapRB=True, crop=False)
	net.setInput(blob)

	# on rentre dans le réseau. Il répond la place des boîtes et des masques correspondant aux détections ainsi que les indices des labels et la confiance associée
	(boxes, masks) = net.forward(["detection_out_final",
		"detection_masks"])

	# il faut looper à l intérieur des détections pour en extraire les infos
	for i in range(0, boxes.shape[2]):
		# l indice (pour le label)
		classID = int(boxes[0, 0, i, 1])
		# la confiance associée à la détection
		confidence = boxes[0, 0, i, 2]

		# on ne veut garder que les détections avec une confiance supérieure à l'option choisie
		if confidence > args["confidence"]:
			# les coordonnées des boîtes sont réajustées par rapport à la taille de l'image
			(H, W) = color_image.shape[:2]
			box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H])
			# les coordonnées sont extraites
			(startX, startY, endX, endY) = box.astype("int")
			boxW = endX - startX
			boxH = endY - startY

			# extrait le masque et son label
			mask = masks[i, classID]
			# le masque est reproportionné à la taille de l image
			mask = cv2.resize(mask, (boxW, boxH),
				interpolation=cv2.INTER_CUBIC)
			# plus l option est petite, plus les masques "débordent"
			mask = (mask > args["threshold"])

			# extrait la région d'intérêt
			roi = color_image[startY:endY, startX:endX][mask]

			# fait le "fondu de couleur" pour l'overlay
			color = COLORS[classID]
			# changer les valeurs en dessous change l effet "transparence"
			blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

			# écrit le roi dans l'image
			color_image[startY:endY, startX:endX][mask] = blended
			# dessine la boîte sur l'image
			color = [int(c) for c in color]
			cv2.rectangle(color_image, (startX, startY), (endX, endY),
				color, 2)

			# écrit le label et la confiance associée
			text = "{}: {:.2f}".format(LABELS[classID], confidence)
			cv2.putText(color_image, text, (startX, startY - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			# calcule la distance en utilsant les pixels masques de la détection (a des avantages et des défauts...)
			# il vaut donc mieux des masques qui ne débordent pas trop (threshold grand)
			# le 10e-4 vient du "depth scale" utilisé par le capteur
			# j arrondi au cm mais ça n'a pas beaucoup de sens au delà de 3.5 mètre
			# plus la cible est large sur le field of view, meilleure est la mesure
			distance = round(np.mean(depth_image[startY:endY, startX:endX][mask].astype(float))*10e-4, 2)
			# écriture de la distance au dessus de la boîte (color)
			cv2.putText(color_image, str(distance) + ' m', (startX, startY - 20), fnt, 1, color, 2)

	# pour être stackée avec l'image couleur, la depth image doit avoir 3 bandes et les valeurs stretchées
	depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.008), cv2.COLORMAP_JET)
	images = np.hstack((depth_colormap, color_image))

	# écriture du vidéo (attention au choix du fps
	if args["output"] != "" and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 16,
			(images.shape[1], images.shape[0]), True)

	if writer is not None:
		writer.write(images)
