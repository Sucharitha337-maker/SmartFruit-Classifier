import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import string

from mlxtend.plotting import plot_decision_regions
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.multiclass import unique_labels
from sklearn import metrics
from sklearn.svm import SVC

# Step 1: Load the CSV data into a DataFrame
df = pd.read_csv('../fruit_all/GA_P_num.csv')


# Identify categorical columns (replace 'categorical_column' with actual column names)
categorical_columns = ['Label']

# Separate features and target variable (if applicable)
#X = df_encoded(columns=['Area','Mean','StdDev','Mode','Min','Max','X','Y','XM','YM','Perim.','BX','BY','Width','Height','Major','Minor','Angle','Circ.','Feret','Median','%Area','Slice','FeretX','FeretY','FeretAngle','MinFeret','AR','Round','Solidity','MinThr','MaxThr'])  # Features
X = df.drop(columns=['Label'])
y = pd.factorize(df['Label'])[0]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Perform PCA with desired number of components
pca = PCA(n_components=3)  # You can adjust the number of components as needed
X_pca = pca.fit_transform(X_scaled)

# Step 4: Visualize the principal components in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


# Define colors for each category
class_colors = {0: 'blue', 1:'red'}  # Add more colors as needed
# Create a list of colors based on the target variable
color_list = [class_colors[label] if label in class_colors else 'gray' for label in y]

ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],c=color_list,alpha=0.5)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('PCA 3D Scatter Plot')
plt.show()



X_train, X_test, y_train, y_test = train_test_split(X_pca, df['Label'], test_size=0.25, random_state=42)

# Step 5: Train KNN model
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with KNN:", accuracy)


from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(set(y_test)))
plt.xticks(tick_marks, sorted(set(y_test)), rotation=45)
plt.yticks(tick_marks, sorted(set(y_test)))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')

# Add text annotations
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()