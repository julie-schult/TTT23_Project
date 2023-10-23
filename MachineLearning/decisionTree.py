from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import csv
import cv2 as cv
import numpy as np
from pathlib import Path
import sys
print(Path.cwd())
sys.path.insert(0, str(Path.cwd() / "image_normalization"))
from image_norm import setGrayToBlack, paddImage
from pymage_size import get_image_size

def readImages(csvPath):

    filenames = []
    roi = []
    max_shape = [0, 0]
    with open(csvPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                path = Path.cwd() / row[0]
                filenames.append(str(path))
                img_size = get_image_size(str(path)).get_dimensions()[::-1]
                if img_size[0] > max_shape[0]:
                    max_shape[0] = img_size[0]
                if img_size[1] > max_shape[1]:
                    max_shape[1] = img_size[1]

                roi.append([float(row[1]), float(row[2]), float(row[3])])

    y = np.zeros((len(filenames), 3), float)
    images = np.zeros((len(filenames), max_shape[0], max_shape[1]), dtype=np.uint8) 
    for i, filepath in enumerate(filenames):
        img = cv.imread(str(filepath))
        # cv.imshow("img1", img)
        img_black = setGrayToBlack(img, threshold=150)
        # cv.imshow("img_gray_black", img)

        img_gray = cv.cvtColor(img_black, cv.COLOR_BGR2GRAY)
        # cv.imshow("img_gray", img)
        img_pad = paddImage(img_gray, max_shape)
        # cv.imshow("pad", img)
        images[i] = img_pad
        y[i] = np.array(roi[i])
    
    return images, y



def normalizeY(y):
    return (y-np.amin(y))/(np.amax(y) - np.amin(y))




    
images, y = readImages(r"C:\Users\sondr\OneDrive\Dokumenter\a\TTT23\TTT23_Project\Rawdata_values.csv")
y = normalizeY(y)
X = images.reshape((images.shape[0], images.shape[1]*images.shape[2])) # Flatten images


from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

# Create a decision tree regressor
regressor = DecisionTreeRegressor()

# Create a MultiOutputRegressor
multi_output_regressor = MultiOutputRegressor(regressor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
multi_output_regressor.fit(X_train, y_train)

# Make predictions

predictions = multi_output_regressor.predict(X_test)


diff = np.zeros(3, dtype=float)
tot_diff = 0
i = 0

print("Imag results: ROI1,\t ROI2,\t ROI3")
for test, pred in zip(y_test, predictions):
    print(f"img{i} actual: {test}, avg: {np.sum(test)/3}")
    print(f"img{i} predic: {pred}, avg: {np.sum(pred)/3}, diff: {test - pred}, tot diff: {np.sum(test-pred)}")
    print()
    l_diff = np.abs(test-pred)
    diff += l_diff
    tot_diff += np.sum(l_diff)/3
    i+=1

print(f"Total difference: {tot_diff/len(y_test)}")
print(f"Average local diff: {diff/len(y_test)}")
