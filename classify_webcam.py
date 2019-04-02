import pickle
import string
import cv2

from sklearn.cluster import MiniBatchKMeans

from preprocessing_surf import cluster_features
from surf_image_processing import func


with open('./classifiers.model', 'rb') as models_file:
    models = pickle.load(models_file)

symbols = list(string.ascii_uppercase)
symbols.extend(['del', 'nothing', 'space'])

clf = models['svm']

cap = cv2.VideoCapture(0)

while(True):
	ret, img = cap.read()
	img = cv2.flip(img, 1)
	cv2.imshow("original",img)
	axis_length = min(img.shape[0], img.shape[1])
	diff = abs(img.shape[1] - img.shape[0])
	if img.shape[0] < img.shape[1]:
	    new_img = img[:, diff//2:img.shape[1]-diff//2]
	else:
	    new_img = img[diff//2:img.shape[0]-diff//2, :]

	resized_img = cv2.resize(new_img, (200, 200))

	image_data = cv2.imencode('.jpg', resized_img)[1].tostring()
	with open('input.jpg', 'wb') as image_file:
	    image_file.write(image_data)

	img_des = func(resized_img)
	try:
		X, cluster_model = cluster_features([img_des], range(1), MiniBatchKMeans(n_clusters=150))
		#print(cluster_model)
		y_pred = clf.predict(X)

		print("\n\nPredicted symbol:")
		print(symbols[int(y_pred)])
	except ValueError:
		print("less features")

	if cv2.waitKey(2500) == ord('q'):
		break

# Following line should... <-- This should work fine now
cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
