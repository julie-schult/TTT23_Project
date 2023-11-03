from sklearn import datasets
from sklearn.model_selection import train_test_split
import csv
import cv2 as cv
import numpy as np
from pathlib import Path
import sys
print(Path.cwd())
sys.path.insert(0, str(Path.cwd() / "TTT23_Project"))
from image_normalization.image_norm import setGrayToBlack, paddImage, rotate, rotate_same_dim
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
                path = Path.cwd() / "TTT23_Project" / row[0]
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
        # cv.imshow("img_gray_black", img_black)

        img_gray = cv.cvtColor(img_black, cv.COLOR_BGR2GRAY)
        # cv.imshow("img_gray", img_gray)
        img_pad = paddImage(img_gray, max_shape)
        # cv.imshow("pad", img_pad)
        images[i] = img_pad
        y[i] = np.array(roi[i])
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # return
    return images, y


def augment_train(X, Y):

    numAugment = np.random.randint(3, 11, size=(len(X)))
    X_out = np.zeros((len(X) + np.sum(numAugment), X.shape[1], X.shape[2]), dtype=X.dtype)
    Y_out = np.zeros((len(Y) + np.sum(numAugment), Y.shape[1]), dtype=Y.dtype)

    out_index = 0

    for i in range(len(X)):
        #augment a random number of times inbetween 3 and 10 a random degree number between -30 and 30 degrees
        X_out[out_index] = X[i]
        Y_out[out_index] = Y[i]
        out_index += 1
        for k in range(numAugment[i]):
            rand_rot = np.random.randint(-30, 30)
            aug_image = rotate_same_dim(X[i], rand_rot)
            X_out[out_index] = aug_image
            Y_out[out_index] = Y[i]
            out_index += 1
    
    return X_out, Y_out







def normalizeY(y):
    log_y = np.log(y)

    return (log_y-np.amin(log_y))/(np.amax(log_y) - np.amin(log_y)), np.amax(log_y), np.amin(log_y)

def deNormalizeYresults(y, maks, min):
    return np.exp(y*(maks - min) + min)


    
images, y = readImages(r"C:\Users\sondr\OneDrive\Dokumenter\a\TTT23\TTT23_Project\Rawdata_values.csv")

# y, maks_y, min_y = normalizeY(y)
# bins = np.zeros(11, int)
# for a in y.flatten():
#     bins[int(a*10)] = bins[int(a*10)] + 1

# print(bins)


# X = images.reshape((images.shape[0], images.shape[1]*images.shape[2])) # Flatten images


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# Create a decision tree regressor
# regressor = DecisionTreeRegressor()
print("regressor generated")
regressor = RandomForestRegressor(n_estimators=100, max_features="sqrt")

# Create a MultiOutputRegressor
multi_output_regressor = MultiOutputRegressor(regressor)

#Split train and test date
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2, random_state=41)

print(f"Starting augmentation of data {X_train.shape}, {y_train.shape}")
X_train_augmented, y_train = augment_train(X_train, y_train)
print(f"Done with augmentation of data {X_train_augmented.shape}, {y_train.shape}")

#flatten images:
X_train = X_train_augmented.reshape((X_train_augmented.shape[0], X_train_augmented.shape[1]*X_train_augmented.shape[2])) # Flatten images

print("Fitting data:")
print(X_train.shape)
multi_output_regressor.fit(X_train, y_train)

# Make predictions
print("Predicting data:")
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
predictions = multi_output_regressor.predict(X_test)


diff = np.zeros(3, dtype=float)
tot_diff = 0
i = 0
diffs = np.zeros(predictions.shape, float)
print("Imag results: ROI1,\t ROI2,\t ROI3")
for test, pred in zip(y_test, predictions):
    d_test = test
    d_pred = pred
    # d_test = deNormalizeYresults(test, maks_y, min_y)
    # d_pred = deNormalizeYresults(pred, maks_y, min_y)


    print(f"img{i} actual: {d_test}, avg: {np.sum(test)/3}")
    print(f"img{i} predic: {d_pred}, avg: {np.sum(pred)/3}, diff: {abs(d_test - d_pred)}, tot diff: {np.sum(test-pred)}")
    print()
    l_diff = np.abs(d_test-d_pred)
    diffs[i] = l_diff
    diff += l_diff
    tot_diff += np.sum(l_diff)/3
    i+=1

print(f"Total difference: {tot_diff/len(y_test)}")
print(f"Average local diff: {diff/len(y_test)}")
print(f"Variance  of diffs: {np.var(diffs, axis=0)}")
