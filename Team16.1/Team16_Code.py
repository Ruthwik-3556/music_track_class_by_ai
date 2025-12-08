#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import random
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn imports
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Oversampling
from imblearn.over_sampling import SMOTE

# ML Libraries
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Encoding
from category_encoders import TargetEncoder

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import warnings
warnings.filterwarnings('ignore')


# In[1]:


# Set visualization style
plt.style.use('ggplot')  # Modern and clean style
sns.set_palette('pastel')  # Soft color palette
# %matplotlib inline


# In[2]:


# Load datasets (replace with your actual file paths)
train_path = "train.csv"  
test_path = "test.csv"    


# In[3]:


try:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print("✅ Data loaded successfully")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    raise


# In[4]:


print("\n=== Data Overview ===")
print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")
print("\nTraining data sample:")
# display(train_df.head(3))


# In[5]:


print(train_df.info())
print(test_df.info())


# In[6]:


train_df.nunique()


# In[7]:


print("Removing Duplicates:")
print("Duplicate values in train_df: ",train_df.duplicated().sum())
if(train_df.duplicated().sum()>0):
    train_df.drop_duplicates(inplace=True)
    train_df.reset_index(drop=True,inplace=True)


# In[8]:


numerical_features = train_df.select_dtypes(include=np.number).columns.tolist()
categorical_features = ['Artist Name', 'Track Name']
print("\nNumerical features:", numerical_features)
print("Categorical features:", categorical_features)


# In[9]:


# Print number of missing values in each feature
print("\nMissing values in train_df:")
print(train_df.isnull().sum())
print("\nMissing values in test_df:")
print(test_df.isnull().sum())


# In[10]:


#filling the missing values for numerical features 
from sklearn.impute import SimpleImputer
# Impute missing values
target="Class"
train_df = train_df.copy()
test_df = test_df.copy()
num_feats_not_target = [col for col in numerical_features if col != target]
imputer = SimpleImputer(strategy="median")
train_df[numerical_features] = imputer.fit_transform(train_df[numerical_features])
test_df[num_feats_not_target]=imputer.fit_transform(test_df[num_feats_not_target])


# In[11]:


print(f"imputation results::")
print(f"Train missing values after imputation:\n{train_df[numerical_features].isnull().sum()}")
print(f"Test missing values after imputation:\n{test_df[num_feats_not_target].isnull().sum()}")


# In[12]:


print("\n=== Starting Exploratory Data Analysis ===")

# 5.1 Genre Distribution
plt.figure(figsize=(14,6))
ax = sns.countplot(data=train_df, x='Class', order=train_df['Class'].value_counts().index)

plt.title('Distribution of Music Genres', fontsize=16, pad=20)
plt.xlabel('Genre', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, ha='right')

# Add percentage labels
total = len(train_df)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width()/2., height + 50,
            f'{height/total:.1%}',
            ha='center', fontsize=10)

plt.tight_layout()
plt.show()


# In[13]:


# 5.2 Feature Distributions of numerical features
print("\n=== Feature Distributions ===")

# Create subplots for numerical features
n_cols = 3
n_rows = int(np.ceil(len(numerical_features) / n_cols))

# Boxplots
plt.figure(figsize=(18, 5*n_rows))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=train_df[col], color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel('')
plt.tight_layout()
plt.suptitle('Numerical Features - Boxplots train_df', y=1.02, fontsize=16)
plt.show()

n_cols = 3
n_rows = int(np.ceil(len(numerical_features) / n_cols))
print("**"*25,"Test Boxplot","**"*25)
# Boxplots
plt.figure(figsize=(18, 5*n_rows))
for i, col in enumerate(num_feats_not_target, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=test_df[col], color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel('')
plt.tight_layout()
plt.suptitle('Numerical Features - Boxplots train_df', y=1.02, fontsize=16)
plt.show()


# In[14]:


#Handling outliers using z-score

#Replace outliers using Z-score with median
continuous_features = ['Popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
                       'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                       'valence', 'tempo', 'duration_in min/ms', 'time_signature']

z_scores = np.abs((train_df[continuous_features] - train_df[continuous_features].mean()) /
                  train_df[continuous_features].std())

for feature in continuous_features:
    median_val = train_df[feature].median()
    train_df.loc[z_scores[feature] > 3, feature] = median_val


# In[15]:


# 5.2 Feature Distributions
print("\n=== Feature Distributions ===")

# Create subplots for numerical features
n_cols = 3
n_rows = int(np.ceil(len(numerical_features) / n_cols))

# Boxplots
plt.figure(figsize=(18, 5*n_rows))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=train_df[col], color='skyblue')
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel('')
plt.tight_layout()
plt.suptitle('Numerical Features - Boxplots', y=1.02, fontsize=16)
plt.show()


# In[16]:


# Histograms
plt.figure(figsize=(18, 5*n_rows))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(train_df[col], bins=30, kde=True, color='teal')
    plt.title(f'Distribution of {col}', fontsize=12)
    plt.xlabel('')
plt.tight_layout()
plt.suptitle('Numerical Features - Histograms', y=1.02, fontsize=16)
plt.show()


# In[17]:


# 5.3 Correlation Analysis
print("\n=== Correlation Analysis ===")

# Correlation with target
corr_with_target = train_df[numerical_features ].corr()[['Class']].sort_values('Class', ascending=False)

plt.figure(figsize=(8,10))
sns.heatmap(corr_with_target, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation with Genre', fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# Full correlation matrix
plt.figure(figsize=(14,12))
corr_matrix = train_df[numerical_features ].corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, fmt=".2f",cmap='coolwarm')
plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
plt.tight_layout()
plt.show()


# In[18]:


# 5.4 Feature Relationships by Genre
print("\n=== Feature Relationships by Genre ===")

plt.figure(figsize=(18, 5*n_rows))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x='Class', y=col, data=train_df)
    plt.title(f'{col} by Genre', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('')
plt.tight_layout()
plt.suptitle('Numerical Features by Genre', y=1.02, fontsize=16)
plt.show()


# In[19]:


# 5.5 Pairplot of Key Features
print("\n=== Pairplot of Key Features ===")

# Select important features based on correlation
key_features = ['danceability', 'energy', 'acousticness', 'valence', 'tempo', 'Class']
key_features = [f for f in key_features if f in train_df.columns]

# Sample data if large
plot_data = train_df.sample(1000) if len(train_df) > 1000 else train_df

sns.pairplot(plot_data[key_features], hue='Class', diag_kind='kde', plot_kws={'alpha':0.6})
plt.suptitle('Relationships Between Key Features by Genre', y=1.02)
plt.tight_layout()
plt.show()


# In[20]:


#multivariate analysis of numerical features before encoding categorical features
# Now we have Define a function for variance_inflation_factor(VIF)
from statsmodels.stats.outliers_influence import variance_inflation_factor
def vif_calc(X):

    # Now Calculate the VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif.sort_values(by = ['VIF'], ascending=False ))


# In[21]:


# Select only numeric columns and drop any that contain NaN or non-numeric values
train_df_numeric = train_df.select_dtypes(include=[np.number]).dropna()

# Then call your function
vif_calc(train_df_numeric)


# In[22]:


# feature engineering and encoding some  categorical features
y=train_df.iloc[:,-1]
X=train_df.copy()
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Frequency encode Track Name
track_freq = X["Track Name"].value_counts().to_dict()
X["Track_Freq"] = X["Track Name"].map(track_freq)

# Target encode Track Name
track_target_encoder = TargetEncoder()
X["Track_Name_TE"] = track_target_encoder.fit_transform(X["Track Name"], y_encoded)

# Drop original Track Name
X = X.drop("Track Name", axis=1)

# Target encode Artist Name
artist_target_encoder = TargetEncoder()
X["Artist Name"] = artist_target_encoder.fit_transform(X["Artist Name"], y_encoded)


# In[23]:


X


# In[24]:


X.isnull().sum()


# In[25]:


X=X.drop("Class",axis=1)


# In[26]:


# Standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[27]:


X_preprocessed_df = pd.DataFrame(X_scaled, columns=X.columns)
X_preprocessed_df["Target"] = y_encoded
# X_preprocessed_df.to_csv("preprocessed_outlier_median_data.csv", index=False)
print("✅ Preprocessing done.")


# In[28]:


# # Select only numeric columns and drop any that contain NaN or non-numeric values
# X_numeric = X.select_dtypes(include=[np.number]).dropna()

# # Then call your function
# vif_calc(X_numeric)


# In[29]:


# X_numeric


# In[30]:


# X_scaled = scaler.fit_transform(X)


# In[31]:


# X_preprocessed_df = pd.DataFrame(X_scaled, columns=X.columns)
# X_preprocessed_df["Target"] = y_encoded
# X_preprocessed_df.to_csv("preprocessed_test.csv", index=False)
# print("✅ Preprocessing done. Saved to preprocessed_test.csv.")


# In[32]:


test_df.isnull().sum()


# In[33]:


# --- Frequency Encode 'Track Name' ---
test_df["Track_Freq"] = test_df["Track Name"].map(track_freq)

# --- Target Encode using pre-fit encoders from training ---
test_df["Track_Name_TE"] = track_target_encoder.transform(test_df["Track Name"])
test_df["Artist Name"] = artist_target_encoder.transform(test_df["Artist Name"])

# --- Drop original 'Track Name' ---
test_df = test_df.drop("Track Name", axis=1)


# In[34]:


X_test=test_df


# In[35]:


missing = X_test["Track_Freq"].isna().sum()
print(f" Unseen Track Names in test(not present in train): {missing}")


# In[36]:


X_test.isnull().sum()


# In[37]:


#filling the missing values in Track_Freq
median_freq = np.median(list(track_freq.values()))
X_test["Track_Freq"].fillna(median_freq)


# In[38]:


#no of missing values after filling
X_test.isnull().sum()


# In[39]:


X_test_scaled = scaler.fit_transform(X_test)


# In[40]:


# X_test_preprocessed_df = pd.DataFrame(X_test_scaled, columns=X.columns)
# X_test_preprocessed_df.to_csv("preprocessed_test.csv", index=False)
# print("✅ Preprocessing done. Saved to preprocessed_test.csv.")


# In[42]:


X_preprocessed_df


# In[43]:


# Fix seeds
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

df=X_preprocessed_df
X_scaled=df.drop(["Target"],axis=1)
y_cleaned = df["Target"]
le=LabelEncoder()
le.fit(y_cleaned)
print()


# In[44]:


way=[]
best_acc=[]
precision=[]
recall=[]
f1_score=[]


# In[45]:


# Define nn model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x

# Stratified K-Fold setup
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
fold = 1
best_overall_acc = 0
best_preds, best_labels = None, None

for train_idx, val_idx in skf.split(X_scaled, y_cleaned):
    print(f"\n--- Fold {fold} ---")
    x_train_fold, x_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train_fold, y_val_fold = y_cleaned.iloc[train_idx], y_cleaned.iloc[val_idx]

    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_fold, y_train_fold)

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_res), y=y_train_res)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train_res.values, dtype=torch.float32), torch.tensor(y_train_res.values, dtype=torch.long)), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val_fold.values, dtype=torch.float32), torch.tensor(y_val_fold.values, dtype=torch.long)), batch_size=64)

    model = SimpleNN(input_dim=X_scaled.shape[1], output_dim=len(np.unique(y_cleaned))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=25)

    best_fold_acc = 0
    best_fold_preds, best_fold_labels = None, None

    for epoch in range(25):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                all_labels.extend(yb.numpy())

        acc = np.mean(np.array(all_preds) == np.array(all_labels))
        if acc > best_fold_acc:
            best_fold_acc = acc
            best_fold_preds = all_preds
            best_fold_labels = all_labels

        print(f"Epoch {epoch+1:2d} | Fold {fold} | Val Accuracy: {acc:.4f}")

    if best_fold_acc > best_overall_acc:
        best_overall_acc = best_fold_acc
        best_preds = best_fold_preds
        best_labels = best_fold_labels

    fold += 1
# target_names = sorted(y_cleaned.unique().tolist())
way.append('NeuralNetwork (sigmoid)')
print("\nBest Validation Accuracy across all folds:", round(best_overall_acc, 4))
best_acc.append(round(best_overall_acc, 4))
print("\nClassification Report:")
target_names = [str(cls) for cls in le.classes_]
print(classification_report(best_labels, best_preds, target_names=target_names))

report = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)
# For macro-average scores
precision.append(round(report['macro avg']['precision'],4))
recall.append(round(report['macro avg']['recall'],4))
f1_score.append(round(report['macro avg']['f1-score'],4))

cm = confusion_matrix(best_labels, best_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[46]:


# Stratified K-Fold setup for LR
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_overall_acc = 0
best_preds, best_labels = None, None

# Training loop with Stratified K-Fold for LR
fold = 1
for train_idx, val_idx in skf.split(X_scaled, y_cleaned):
    print(f"\n--- Fold {fold} ---")
    x_train_fold, x_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train_fold, y_val_fold = y_cleaned.iloc[train_idx], y_cleaned.iloc[val_idx]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_fold, y_train_fold)

    # Compute class weights for the training set
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_res), y=y_train_res)
    class_weights_dict = dict(zip(np.unique(y_train_res), class_weights))

    # Initialize and train Logistic Regression model
    model = LogisticRegression(max_iter=1000, class_weight=class_weights_dict, solver='lbfgs', multi_class='multinomial', random_state=42)
    model.fit(x_train_res, y_train_res)

    # Evaluate the model on the validation set
    val_preds = model.predict(x_val_fold)
    val_accuracy = np.mean(val_preds == y_val_fold)
    print(f"Validation Accuracy for Fold {fold}: {val_accuracy:.4f}")

    # Track best performing fold
    if val_accuracy > best_overall_acc:
        best_overall_acc = val_accuracy
        best_preds = val_preds
        best_labels = y_val_fold

    fold += 1

# Print best validation accuracy across all 
way.append('Logistic Regression')
print("\nBest Validation Accuracy across all folds:", round(best_overall_acc, 4))
best_acc.append(round(best_overall_acc, 4))
# Print the classification report
print("\nClassification Report:")
# target_names = sorted(y_cleaned.unique().tolist())
target_names = [str(cls) for cls in le.classes_]
print(classification_report(best_labels, best_preds, target_names=target_names))

report = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)
# For macro-average scores
precision.append(round(report['macro avg']['precision'],4))
recall.append(round(report['macro avg']['recall'],4))
f1_score.append(round(report['macro avg']['f1-score'],4))

cm = confusion_matrix(best_labels, best_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[47]:


##  Naive Bayes Classifier
# ------------------------
print("\n===== Naive Bayes Classifier =====")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold = 1
best_overall_acc = 0
best_preds, best_labels = None, None

for train_idx, val_idx in skf.split(X_scaled, y_cleaned):
    print(f"\n--- Fold {fold} ---")
    x_train_fold, x_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train_fold, y_val_fold = y_cleaned.iloc[train_idx], y_cleaned.iloc[val_idx]

    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_fold, y_train_fold)

    model = GaussianNB()
    model.fit(x_train_res, y_train_res)

    val_preds = model.predict(x_val_fold)
    acc = np.mean(val_preds == y_val_fold)
    print(f"Fold {fold} | Validation Accuracy: {acc:.4f}")

    if acc > best_overall_acc:
        best_overall_acc = acc
        best_preds = val_preds
        best_labels = y_val_fold

    fold += 1
way.append('Naive Bayes')
print("\nBest Validation Accuracy:", round(best_overall_acc, 4))
best_acc.append(round(best_overall_acc, 4))
print("\nClassification Report:")
target_names = [str(cls) for cls in le.classes_]
print(classification_report(best_labels, best_preds, target_names=target_names))

report = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)
# For macro-average scores
precision.append(round(report['macro avg']['precision'],4))
recall.append(round(report['macro avg']['recall'],4))
f1_score.append(round(report['macro avg']['f1-score'],4))

# Confusion Matrix
cm = confusion_matrix(best_labels, best_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[55]:


# Stratified K-Fold setup for Random Forest
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_overall_acc = 0
best_preds, best_labels = None, None

# Training loop with Stratified K-Fold for Random Forest
fold = 1
for train_idx, val_idx in skf.split(X_scaled, y_cleaned):
    print(f"\n--- Fold {fold} ---")
    x_train_fold, x_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train_fold, y_val_fold = y_cleaned.iloc[train_idx], y_cleaned.iloc[val_idx]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_fold, y_train_fold)

    # Compute class weights for the training set
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_res), y=y_train_res)
    class_weights_dict = dict(zip(np.unique(y_train_res), class_weights))

    # Initialize and train Random Forest model
    model = RandomForestClassifier(n_estimators=100, class_weight=class_weights_dict, random_state=42)
    model.fit(x_train_res, y_train_res)

    # Evaluate the model on the validation set
    val_preds = model.predict(x_val_fold)
    val_accuracy = np.mean(val_preds == y_val_fold)
    print(f"Validation Accuracy for Fold {fold}: {val_accuracy:.4f}")

    # Track best performing fold
    if val_accuracy > best_overall_acc:
        best_model_cls=model
        best_overall_acc = val_accuracy
        best_preds = val_preds
        best_labels = y_val_fold

    fold += 1

# Print best validation accuracy across all folds
way.append('Random Forest')
print("\nBest Validation Accuracy across all folds:", round(best_overall_acc, 4))
best_acc.append(round(best_overall_acc, 4))
# Print the classification report
print("\nClassification Report:")
print(classification_report(best_labels, best_preds, target_names=target_names))

report = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)
# For macro-average scores
precision.append(round(report['macro avg']['precision'],4))
recall.append(round(report['macro avg']['recall'],4))
f1_score.append(round(report['macro avg']['f1-score'],4))

# Compute and print confusion matrix
cm = confusion_matrix(best_labels, best_preds)

# Optional: Visualize the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[49]:


# Stratified K-Fold setup for Decision Tree
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_overall_acc = 0
best_preds, best_labels = None, None

# Training loop with Stratified K-Fold for Decision Tree
fold = 1
for train_idx, val_idx in skf.split(X_scaled, y_cleaned):
    print(f"\n--- Fold {fold} ---")
    x_train_fold, x_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train_fold, y_val_fold = y_cleaned.iloc[train_idx], y_cleaned.iloc[val_idx]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_fold, y_train_fold)

    # Compute class weights for the training set
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_res), y=y_train_res)
    class_weights_dict = dict(zip(np.unique(y_train_res), class_weights))

    # Initialize and train Decision Tree model
    model = DecisionTreeClassifier(class_weight=class_weights_dict, random_state=42)
    model.fit(x_train_res, y_train_res)

    # Evaluate the model on the validation set
    val_preds = model.predict(x_val_fold)
    val_accuracy = np.mean(val_preds == y_val_fold)
    print(f"Validation Accuracy for Fold {fold}: {val_accuracy:.4f}")

    # Track best performing fold
    if val_accuracy > best_overall_acc:
        best_overall_acc = val_accuracy
        best_preds = val_preds
        best_labels = y_val_fold
        dec_tree=model
    fold += 1

# Append Decision Tree model to 'way' and its best accuracy to 'best_acc'
way.append('Decision Tree')
best_acc.append(round(best_overall_acc, 4))

# Print best validation accuracy across all folds
print("\nBest Validation Accuracy across all folds:", round(best_overall_acc, 4))

# Print the classification report
print("\nClassification Report:")
target_names = [str(cls) for cls in np.unique(y_cleaned)]  # Adjust target names if needed
print(classification_report(best_labels, best_preds, target_names=target_names))

report = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)

# For macro-average scores
precision.append(round(report['macro avg']['precision'], 4))
recall.append(round(report['macro avg']['recall'], 4))
f1_score.append(round(report['macro avg']['f1-score'], 4))

# Compute and print confusion matrix
cm = confusion_matrix(best_labels, best_preds)

# Optional: Visualize the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[50]:


#SVM
# Stratified K-Fold setup for SVM
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_overall_acc = 0
best_preds, best_labels = None, None

# Training loop
fold = 1
for train_idx, val_idx in skf.split(X_scaled, y_cleaned):
    print(f"\n--- Fold {fold} ---")
    x_train_fold, x_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train_fold, y_val_fold = y_cleaned.iloc[train_idx], y_cleaned.iloc[val_idx]

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_fold, y_train_fold)

    # Compute sample weights from class weights
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_res), y=y_train_res)
    class_weights_dict = dict(zip(np.unique(y_train_res), class_weights))
    sample_weights = np.array([class_weights_dict[label] for label in y_train_res])

    # Initialize SVM model with RBF kernel
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

    # Train the model
    model.fit(x_train_res, y_train_res, sample_weight=sample_weights)

    # Make predictions
    val_preds = model.predict(x_val_fold)
    val_accuracy = np.mean(val_preds == y_val_fold)
    print(f"Validation Accuracy for Fold {fold}: {val_accuracy:.4f}")

    if val_accuracy > best_overall_acc:
        best_overall_acc = val_accuracy
        best_preds = val_preds
        best_labels = y_val_fold

    fold += 1

# Final metrics
way.append('SVM Classifier rbf kernel')
print("\nBest Validation Accuracy across all folds:", round(best_overall_acc, 4))
best_acc.append(round(best_overall_acc, 4))

# Print classification report
print("\nClassification Report:")
print(classification_report(best_labels, best_preds, target_names=target_names))

report = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)
# For macro-average scores
precision.append(round(report['macro avg']['precision'],4))
recall.append(round(report['macro avg']['recall'],4))
f1_score.append(round(report['macro avg']['f1-score'],4))
# Confusion Matrix
cm = confusion_matrix(best_labels, best_preds)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - SVM')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[51]:


# KNN
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_overall_acc = 0
best_k = None
best_model = None
best_preds, best_labels = None, None

# Training loop
fold = 1
for train_idx, val_idx in skf.split(X_scaled, y_cleaned):  # Using y_cleaned as the target variable
    print(f"\n--- Fold {fold} ---")
    x_train_fold, x_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train_fold, y_val_fold = y_cleaned.iloc[train_idx], y_cleaned.iloc[val_idx]

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_fold, y_train_fold)

    # Try different k values
    for k in range(5, 21):  # Try k from 5 to 20
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train_res, y_train_res)  # Removed sample_weight as KNN doesn't support it
        val_preds = knn.predict(x_val_fold)
        val_accuracy = np.mean(val_preds == y_val_fold)
        print(f"K = {k}, Validation Accuracy = {val_accuracy:.4f}")

        if val_accuracy > best_overall_acc:
            best_overall_acc = val_accuracy
            best_k = k
            best_model = knn
            best_preds = val_preds
            best_labels = y_val_fold

    fold += 1

# Final evaluation
print("\nKNN Model Evaluation:")
print(f"Best k: {best_k}")
way.append(f'KNN Model with k={best_k}')
print(f"Best Validation Accuracy: {best_overall_acc:.4f}")
best_acc.append(round(best_overall_acc, 4))
print("Classification Report:")
print(classification_report(best_labels, best_preds))

report = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)
# For macro-average scores
precision.append(round(report['macro avg']['precision'],4))
recall.append(round(report['macro avg']['recall'],4))
f1_score.append(round(report['macro avg']['f1-score'],4))

print("Confusion Matrix:")
cm = confusion_matrix(best_labels, best_preds)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_cleaned), yticklabels=np.unique(y_cleaned))
plt.title('Confusion Matrix - KNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[52]:


# PERCEPTRON WITH POLYNOMIAL FEATURES
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

best_overall_acc = 0
best_preds, best_labels = None, None
best_degree = None
best_model = None

fold = 1
for train_idx, val_idx in skf.split(X_scaled, y_cleaned):
    print(f"\n--- Fold {fold} ---")
    x_train_fold, x_val_fold = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train_fold, y_val_fold = y_cleaned.iloc[train_idx], y_cleaned.iloc[val_idx]

    # SMOTE for imbalance
    smote = SMOTE(random_state=42)
    x_train_res, y_train_res = smote.fit_resample(x_train_fold, y_train_fold)

    # Try polynomial degrees
    for degree in range(1, 4):  # Degrees 1 to 3
        poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
        x_train_poly = poly.fit_transform(x_train_res)
        x_val_poly = poly.transform(x_val_fold)

        # Train Perceptron
        perceptron = Perceptron(max_iter=1000, eta0=1.0, class_weight='balanced', random_state=42)
        perceptron.fit(x_train_poly, y_train_res)

        # Predict
        val_preds = perceptron.predict(x_val_poly)
        val_accuracy = np.mean(val_preds == y_val_fold)
        print(f"Degree = {degree}, Fold {fold}, Validation Accuracy = {val_accuracy:.4f}")

        if val_accuracy > best_overall_acc:
            best_overall_acc = val_accuracy
            best_preds = val_preds
            best_labels = y_val_fold
            best_degree = degree
            best_model = perceptron

    fold += 1

# Final metrics
way.append(f'Perceptron PolynomialDeg={best_degree}')
print("\nBest Validation Accuracy across all folds:", round(best_overall_acc, 4))
best_acc.append(round(best_overall_acc, 4))
print("\nClassification Report:")
print(classification_report(best_labels, best_preds, target_names=target_names))

report = classification_report(best_labels, best_preds, target_names=target_names, output_dict=True)
# For macro-average scores
precision.append(round(report['macro avg']['precision'],4))
recall.append(round(report['macro avg']['recall'],4))
f1_score.append(round(report['macro avg']['f1-score'],4))

# Confusion Matrix
cm = confusion_matrix(best_labels, best_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - Perceptron')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[53]:


print(way,best_acc)


# In[54]:


results_df = pd.DataFrame({
    'Model': way,
    'Best Validation Accuracy': best_acc,
    'precision':precision,
    'recall':recall,
    'f1_score':f1_score
})

# Optional: sort by accuracy for better presentation
results_df = results_df.sort_values(by='Best Validation Accuracy', ascending=False).reset_index(drop=True)
results_df.index = results_df.index + 1
# Display the DataFrame
print(results_df)


# In[61]:


y_pred = best_model_cls.predict(X_test_scaled)

# Define class names based on submission format
class_names = [
    "Acoustic/Folk_0", "Alt_Music_1", "Blues_2", "Bollywood_3", "Country_4",
    "HipHop_5", "Indie Alt_6", "Instrumental_7", "Metal_8", "Pop_9", "Rock_10"
]

# One-hot encode predictions
submission_df = pd.DataFrame(0, index=np.arange(len(y_pred)), columns=class_names)
for idx, label in enumerate(y_pred):
    submission_df.iloc[idx, label] = 1

# Step 5: Save to CSV
submission_df.to_csv("submission.csv", index=False)


# In[62]:


print("submission.csv Created")


# In[ ]:




