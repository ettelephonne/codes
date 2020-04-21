"bag2vid.py" permet de passer d'un fichier .bag (format d'enregistrement du realsense D415) à des fichiers .avi. Il est important de noter que durant la conversion, l'infgormation de distance est perdue.

"bag2matrices.py" permet de passer d'un fichier .bag (format d'enregistrement du realsense D415) à des matrices au fomat .npy. Cela permet lors de calculs effectués en différé, qu'aucune frame ne soit perdue. 

"matrices2avi_r_cnn.py" fait de la segmentation d'objets sur les matrices préparées à l'étape précédente. La sortie du réseau donne un id, une confiance, une boîte et un masque. Le masque est superposé à l'image de profondeur pour calculer la distance de l'objet. C'est plus représentatif de la "vraie" distance que de viser le milieu de la boîte de détection. Ça prend en compte la forme de l'objet. La distance de l'objet est écrite au dessus de la boîte de détection en m. La précision est de l'odre du centimètre pour les distances inférieures à 3 mètres. De la dizaine de cm jusqu'à 5-7 mètres. La valeur affichée pour les distances plus grandes donnent un bon ordre de grandeur.  


