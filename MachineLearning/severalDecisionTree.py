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
import pandas as pd
import time

def readImages(csvPath):

    filenames = []
    roi = []
    max_shape = [0, 0]
    with open(csvPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
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
        img_black = img
        #img_black = setGrayToBlack(img, threshold=150)
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

    
images, y = readImages(r"C:\Users\sondr\OneDrive\Dokumenter\a\TTT23\TTT23_Project\data.csv")

y = y/(10e5)

X = images.reshape((images.shape[0], images.shape[1]*images.shape[2])) # Flatten images

n_estimators_arr = [100, X.shape[0], 200, int(X.shape[0]/2)] # number of trees # 40 is both best and worst. 
n_estimators_arr = [100, int(X.shape[0]/2), 4, int(X.shape[0]/4), 2] # number of trees # 40 is both best and worst. 
n_estimators = 40

min_samples_split_arr_r = [2] # minimum number of samples needed to split a node #2 best on random forest
min_samples_split_arr_d = [0.1, 6] # minimum number of samples needed to split a node # 0.1 or 6 best on decision tree

min_samples_leaf_arr = [1] # 1 best on random forest, 1 or 0.05 best on normal

criterion = ["squared_error", "friedman_mse", "poisson"] # Function to measure quality of a split # Poissons best on random forest, squared_error best on normal but worse on random forest
criterion_f = "poisson" # Function to measure quality of a split # Poissons best on random forest, squared_error best on normal but worse on random forest
criterion_d = "squared_error" # Function to measure quality of a split # Poissons best on random forest, squared_error best on normal but worse on random forest

max_features = ["sqrt"] #, "log2"] # Log2 best on decision tree, sqrt best on random forest



bootstrap = True

n_jobs = 8

oob_score = False

max_samples_arr = [None, int(X.shape[0]/2)] # These weere best fromtesting



# min_samples_split_arr = [2, 0.5]
# min_samples_leaf_arr = [2, 0.5]

#test of vriterion and max_features

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

out_data = None

data = {
    "n_estimators" :      [],
    "min_samples_split" : [],
    "min_samples_leaf" :  [],
    "criterion" :         [],
    "max_features" :      [],
    "bootstrap" :         [],
    "oob_score" :         [],
    "n_jobs" :            [],
    "max_samples" :       [],
    "mse1":               [],
    "mse2":               [],
    "mse3":               [],
    "mse_tot":            [],
    "random_forest":      [],
    "time":               [],
    "MAE1": [],
    "MAE2": [],
    "MAE3": [],
    "MAE_tot": []
}


k = 0
# num_tests = len(min_samples_split_arr)*len(min_samples_leaf_arr)*len(criterion)*len(max_features)*(len(max_samples_arr)*len(n_estimators_arr) + 1)
num_tests = len(min_samples_split_arr_r)*len(min_samples_leaf_arr)*len(criterion)*len(max_features)*(len(max_samples_arr) + 1)
for min_samples_split in min_samples_split_arr_r:
    for min_samples_leaf in min_samples_leaf_arr:
        for max_feature in max_features:
            for max_samples in max_samples_arr:
                for n_estimators in n_estimators_arr:
                    regressor = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion_f, max_features=max_feature, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, max_samples=max_samples)
                    multi_output_regressor = MultiOutputRegressor(regressor)


                    start = time.time()
                    multi_output_regressor.fit(X_train, y_train)
                    predictions = multi_output_regressor.predict(X_test)
                    end = time.time()

                    mse_per_val = np.sqrt(np.mean((y_test - predictions)**2, axis=0))
                    mse_tot = np.sqrt(np.mean((y_test - predictions)**2))

                    MAE = np.mean(np.abs(y_test - predictions), axis=0)
                    MAE_tot = np.mean(np.abs(y_test - predictions))


                    data["n_estimators"].append(n_estimators)
                    data["min_samples_split"].append(min_samples_split)
                    data["min_samples_leaf"].append(min_samples_leaf)
                    data["criterion"].append(criterion_f)
                    data["max_features"].append(max_feature)
                    data["bootstrap"].append(bootstrap)
                    data["oob_score"].append(oob_score)
                    data["n_jobs"].append(n_jobs)
                    data["max_samples"].append(max_samples)
                    data["mse1"].append(mse_per_val[0])
                    data["mse2"].append(mse_per_val[1])
                    data["mse3"].append(mse_per_val[2])
                    data["mse_tot"].append(mse_tot)                    
                    data["MAE1"].append(MAE[0])
                    data["MAE2"].append(MAE[1])
                    data["MAE3"].append(MAE[2])
                    data["MAE_tot"].append(MAE_tot)                    
                    data["random_forest"].append(True)
                    data["time"].append(end - start)
                    print(f"job {k} of {num_tests}")
                    k+=1
            
            
            # # regressor = DecisionTreeRegressor(criterion=criterion_d, min_samples_split=6, min_samples_leaf=min_samples_leaf, max_features="log2")
            # regressor = DecisionTreeRegressor(criterion="poisson", min_samples_split=0.1, min_samples_leaf=1, max_features="log2")
            # multi_output_regressor = MultiOutputRegressor(regressor)

            # for i in range(100):
            #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            #     start = time.time()
            #     multi_output_regressor.fit(X_train, y_train)
            #     predictions = multi_output_regressor.predict(X_test)
            #     end = time.time()

            #     mse_per_val = np.sqrt(np.mean((y_test - predictions)**2, axis=0))
            #     mse_tot = np.sqrt(np.mean((y_test - predictions)**2))

            #     MAE = np.mean(np.abs(y_test - predictions), axis=0)
            #     MAE_tot = np.mean(np.abs(y_test - predictions))

            #     data["n_estimators"].append(None)
            #     data["min_samples_split"].append(0.1)
            #     data["min_samples_leaf"].append(1)
            #     data["criterion"].append("poisson")
            #     data["max_features"].append("log2")
            #     data["bootstrap"].append(None)
            #     data["oob_score"].append(None)
            #     data["n_jobs"].append(None)
            #     data["max_samples"].append(None)
            #     data["mse1"].append(mse_per_val[0])
            #     data["mse2"].append(mse_per_val[1])
            #     data["mse3"].append(mse_per_val[2])
            #     data["mse_tot"].append(mse_tot)  
            #     data["MAE1"].append(MAE[0])
            #     data["MAE2"].append(MAE[1])
            #     data["MAE3"].append(MAE[2])
            #     data["MAE_tot"].append(MAE_tot)     
            #     data["random_forest"].append(False)
            #     data["time"].append(end - start)
            #     print(f"job {k} of {num_tests}")
            #     # print(data)
            #     k+=1


out_data = pd.DataFrame(data)



print(out_data.loc[np.argmin(out_data["mse1"])])
print(out_data.loc[np.argmin(out_data["mse2"])])
print(out_data.loc[np.argmin(out_data["mse3"])])
print(out_data.loc[np.argmin(out_data["mse_tot"])])
print(out_data.loc[np.argmin(out_data["MAE1"])])
print(out_data.loc[np.argmin(out_data["MAE2"])])
print(out_data.loc[np.argmin(out_data["MAE3"])])
print(out_data.loc[np.argmin(out_data["MAE_tot"])])

# print(np.argmax(out_data["mse1"]))
# print(np.argmax(out_data["mse2"]))
# print(np.argmax(out_data["mse3"]))
# print(np.argmax(out_data["mse_tot"]))
# print(np.argmax(out_data["MAE1"]))
# print(np.argmax(out_data["MAE2"]))
# print(np.argmax(out_data["MAE3"]))
# print(np.argmax(out_data["MAE_tot"]))


# out_data.to_csv(f"result{time.strftime('%d_%m_%H_%M', time.gmtime())}.csv")
# for index, row in out_data.iterrows():
#     print(row)


