from tabulate import tabulate
import pandas as pd
import numpy as np

# df = pd.read_csv(R"C:\Users\sondr\OneDrive\Dokumenter\a\TTT23\result11_11_11_55.csv")
df = pd.read_csv(R"C:\Users\sondr\OneDrive\Dokumenter\a\TTT23\result11_11_14_12.csv")
print(df.loc[np.argmin(df["mse1"])])
print(df.loc[np.argmin(df["mse2"])])
print(df.loc[np.argmin(df["mse3"])])
print(df.loc[np.argmin(df["mse_tot"])])
print(df.loc[np.argmax(df["mse1"])])
print(df.loc[np.argmax(df["mse2"])])
print(df.loc[np.argmax(df["mse3"])])
print(df.loc[np.argmax(df["mse_tot"])])
indecies = [np.argmin(df["mse1"]), np.argmin(df["mse2"]), np.argmin(df["mse3"]), np.argmin(df["mse_tot"]), np.argmax(df["mse1"]), np.argmax(df["mse2"]), np.argmax(df["mse3"]), np.argmax(df["mse_tot"])]

def print_max_mean_avg(df):
    print(np.amax(df["mse1"]), np.amin(df["mse1"]), np.average(df["mse1"]))
    print(np.amax(df["mse2"]), np.amin(df["mse2"]), np.average(df["mse2"]))
    print(np.amax(df["mse3"]), np.amin(df["mse3"]), np.average(df["mse3"]))
    print(np.amax(df["mse_tot"]), np.amin(df["mse_tot"]), np.average(df["mse_tot"]))


df_forest = df.loc[df["random_forest"] == True]
df_decision = df.loc[df["random_forest"] == False]
print_max_mean_avg(df_decision)
df_forest_40 = df_forest.loc[df_forest["n_estimators"] == 40]
print_max_mean_avg(df_forest_40)
df_forest_100 = df_forest.loc[df_forest["n_estimators"] == 100]
print_max_mean_avg(df_forest_100)

df_forest_args100 = df_forest.loc[df_forest["n_estimators"] == 100]


indecies_f = [
    df_forest.index[np.argmin(df_forest["mse1"])],   
    df_forest.index[np.argmin(df_forest["mse2"])],   
    df_forest.index[np.argmin(df_forest["mse3"])],   
    df_forest.index[np.argmin(df_forest["mse_tot"])],   
    df_forest.index[np.argmax(df_forest["mse1"])],   
    df_forest.index[np.argmax(df_forest["mse2"])],   
    df_forest.index[np.argmax(df_forest["mse3"])],   
    df_forest.index[np.argmax(df_forest["mse_tot"])]
    ]  
indecies_f_args = [
    df_forest_args100.index[np.argmin(df_forest_args100["mse1"])],   
    df_forest_args100.index[np.argmin(df_forest_args100["mse2"])],   
    df_forest_args100.index[np.argmin(df_forest_args100["mse3"])],   
    df_forest_args100.index[np.argmin(df_forest_args100["mse_tot"])],   
    df_forest_args100.index[np.argmax(df_forest_args100["mse1"])],   
    df_forest_args100.index[np.argmax(df_forest_args100["mse2"])],   
    df_forest_args100.index[np.argmax(df_forest_args100["mse3"])],   
    df_forest_args100.index[np.argmax(df_forest_args100["mse_tot"])]
    ]  
print(df_forest)
print(df)

print(tabulate(df.loc[indecies], headers='keys', tablefmt='psql'))
print(tabulate(df_forest.loc[indecies_f], headers='keys', tablefmt='psql'))
print(tabulate(df_forest_args100.loc[indecies_f_args], headers='keys', tablefmt='psql'))

one = np.unique([np.argmin(df["mse1"]), np.argmin(df["mse2"]), np.argmin(df["mse3"]), np.argmin(df["mse_tot"])])
two = np.unique([df_forest.index[np.argmin(df_forest["mse1"])], df_forest.index[np.argmin(df_forest["mse2"])], df_forest.index[np.argmin(df_forest["mse3"])], df_forest.index[np.argmin(df_forest["mse_tot"])]])
three = np.unique([df_forest_args100.index[np.argmin(df_forest_args100["mse1"])], df_forest_args100.index[np.argmin(df_forest_args100["mse2"])], df_forest_args100.index[np.argmin(df_forest_args100["mse3"])], df_forest_args100.index[np.argmin(df_forest_args100["mse_tot"])]])

print(one[0])
print(two.dtype)
print(three.dtype)
print(np.concatenate((np.concatenate((one, two)), three)))

mse1 = []
mse2 = []
mse3 = []
mse_tot = []

for a in np.concatenate((np.concatenate((one, two)), three)):
    print(df.loc[a]["n_estimators"], df.loc[a]["min_samples_split"], df.loc[a]["min_samples_leaf"], df.loc[a]["criterion"], df.loc[a]["max_features"], df.loc[a]["max_samples"], df.loc[a]["mse1"], df.loc[a]["mse2"], df.loc[a]["mse3"], df.loc[a]["mse_tot"], df.loc[a]["time"])
    mse1.append(df.loc[a]["mse1"])
    mse2.append(df.loc[a]["mse2"])
    mse3.append(df.loc[a]["mse3"])
    mse_tot.append(df.loc[a]["mse_tot"])


from matplotlib import pyplot as plt
import matplotlib.colors as colors
fig, ax = plt.subplots()
x = np.arange(1, len(mse1) + 1, 1)
l_1, = ax.plot(x, mse1, color="g")
l_1.set_label("RMSE1")
l_2, = ax.plot(x, mse2, color="b")
l_2.set_label("RMSE2")
l_3, = ax.plot(x, mse3, color="r")
l_3.set_label("RMSE3")
l_4, = ax.plot(x, mse_tot, color="y")
l_4.set_label("RMSE tot")
ax.legend(loc='upper right')

plt.show()





res = df["mse1"]/3 + df["mse2"]/3 + df["mse3"]/3

print(df.loc[np.argmin(res)])

