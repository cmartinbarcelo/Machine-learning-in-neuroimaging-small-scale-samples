# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 20:34:21 2025

@author: crist
"""

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, roc_auc_score, confusion_matrix,
                             auc, roc_curve, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import os
from os.path import join
import random
from xgboost import XGBClassifier
import seaborn as sns
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata

# %% OBJECTIVE 1. APPLICATION
def model_hyperparameters(model_name):
    if model_name == "kNN":
        return {
            "model__n_neighbors": [3, 5, 7],
            "model__weights": ["uniform", "distance"]
        }

    elif model_name == "SVM":
        return {
            "model__kernel": ["linear", "rbf"],
            "model__C": [0.1, 1, 10, 100],
            "model__gamma": [0.001, 0.01, 0.1, 1]
        }

    elif model_name == "Random Forest":
        return {
            "model__max_depth": [5, 10],
            "model__min_samples_split": [2, 5],
            "model__n_estimators": [50, 100]
        }

    elif model_name == "XGBoost":
        return {
            "model__n_estimators": [50, 100],
            "model__min_child_weight": [1, 3, 6]
        }
    else:
        raise ValueError(f"Model '{model_name}' no reconocido.")


def normalize_volumes(df, volumes, itv_var):
    df_norm = df.copy()
    for volume in volumes:
        df_norm[volume] = (df_norm[volume] / df_norm[itv_var]) * 100

    return df_norm


def scaling(X_train, X_test):
    scaler = StandardScaler()  # define scaler
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def evaluate_model(y_true, y_pred, y_pred_proba, model, do_plot, o_dir=None, cv=None):

    # AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    auc_var = auc(fpr, tpr)
    con_matrix = confusion_matrix(y_true, y_pred)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        # 'recall': recall_score(y_true, y_pred),
        'specificity': con_matrix[1, 1]/(con_matrix[1, 1]+con_matrix[1, 0]),
        'sensitivity': con_matrix[0, 0]/(con_matrix[0, 0]+con_matrix[0, 1]),
        'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1])
    }
        
    if do_plot==True:
        if y_pred_proba.shape[1] == 1:
            # Solo hay una columna, crear la segunda
            y_pred_proba = np.column_stack(
                [1 - y_pred_proba[:, 0], y_pred_proba[:, 0]])
    
        # CONF MAT
        disp = ConfusionMatrixDisplay(confusion_matrix=con_matrix)
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax)
        plt.title(f"Confusion matrix of {model}", fontsize=20)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.savefig(join(o_dir, f"confusion_matrix_{model}_{cv}.jpg"))
        plt.show()
    
        # Plot
        plt.figure()
        plt.plot(fpr, tpr, "b-",
                 label=f'Model {model} (AUC = {auc_var:.2f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.title(f"ROC curve of {model}")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(join(o_dir, f"AUC_{model}_{cv}.jpg"))
        plt.show()
        return metrics, fpr, tpr, auc_var
    else:
        return metrics, fpr, tpr, auc_var

def plot_feature_importance(importance_df, model_name):
    plt.figure(figsize=(8, 4))
    sns.barplot(importance_df, x="Importance", y="Feature", palette="vlag")
    plt.title(f"{model_name} feature importance")
    plt.tight_layout()
    plt.show()

def kfold_cv_classification(X, y, model_name, model, output_dir, cv):
    # Define pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectFromModel(RandomForestClassifier(random_state=42))),
        ("model", model)
    ])
    
    # Obtain param grid
    param_grid = model_hyperparameters(model_name)

    # Define df and results for 2 experiments
    # First experiment: all data
    X_exp1 = X
    results_exp1 = {"model": [], "fold": [], "accuracy": [], "auc": [], 
               "specificity": [], "sensitivity": [], "precision": [], 
               "selected_features": [], "best_params": []}
    cm_exp1 = np.zeros((2,2))  # empty confusion matrix
    # Second experiment: only neuroimaging data
    X_exp2 = X.drop(["Age", "Sex"], axis=1)
    results_exp2 = {"model": [], "fold": [], "accuracy": [], "auc": [], 
               "specificity": [], "sensitivity": [], "precision": [], 
               "selected_features": [], "best_params": []}
    cm_exp2 = np.zeros((2,2))
        
    for fold_num, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        
        print(f"{model_name}, fold {fold_num}")
        
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        for X_data, results, cm_all in [(X_exp1, results_exp1, cm_exp1), (X_exp2, results_exp2, cm_exp2)]:
            
            X_train, X_test = X_data.iloc[train_idx], X_data.iloc[test_idx]
            # grid search
            grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
            grid.fit(X_train, y_train)
            # obtain best pipeline
            best_pipeline = grid.best_estimator_
            selector = best_pipeline.named_steps["selector"]
            selected_features = X_data.columns[selector.get_support()].tolist()
            # make predictions
            y_pred = best_pipeline.predict(X_test)
            y_prob = best_pipeline.predict_proba(X_test)[:, 1]
            
            cm = confusion_matrix(y_test, y_pred)
            cm_all += cm  # Acummulate folds cm
            
            tn, fp, fn, tp = cm.ravel()
            auc_val = roc_auc_score(y_test, y_prob)
            
            results["model"].append(model_name)
            results["fold"].append(fold_num)
            results["accuracy"].append(accuracy_score(y_test, y_pred))
            results["auc"].append(auc_val)
            results["specificity"].append(tn / (tn + fp))
            results["sensitivity"].append(tp / (tp + fn))
            results["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            results["selected_features"].append(selected_features)
            results["best_params"].append(grid.best_params_)
            
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Experiment 1
    sns.heatmap(cm_exp1, annot=True, fmt='g', cmap='viridis', 
                xticklabels=['0', '1'],
                yticklabels=['0', '1'], ax=axes[0])
    axes[0].set_title(f'{model_name} - Neuroimaging + Demographic data',fontsize=20)
    axes[0].set_ylabel('True Label',fontsize=18)
    axes[0].set_xlabel('Predicted Label',fontsize=18)
    axes[0].tick_params(axis='both', labelsize=16)
    
    # Experiment 2
    sns.heatmap(cm_exp2, annot=True, fmt='g', cmap='viridis', 
                xticklabels=['0', '1'],
                yticklabels=['0', '1'], ax=axes[1])
    axes[1].set_title(f'{model_name} - Neuroimaging data',fontsize=20)
    axes[1].set_ylabel('True Label',fontsize=18)
    axes[1].set_xlabel('Predicted Label',fontsize=18)
    axes[1].tick_params(axis='both', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrices.jpg'), dpi=300)
    plt.show()
            
    return pd.DataFrame(results_exp1), pd.DataFrame(results_exp2)

# %% LOAD DATA
random.seed(123)  # set random seed for reproducibility

wdir = os.getcwd()
bbdd_dir = join(wdir, "BBDD")
dataset = "RBD"

# load data
output_dir = join(wdir, "results_new", dataset)
os.makedirs(output_dir, exist_ok=True)

imputed_df = pd.read_excel(join(bbdd_dir, f"imputed_BBDD_{dataset}.xlsx"))

volume_vars = ['Lateral-Ventricle', 'Thalamus', 'Caudate', 'Putamen', 'Pallidum', 'Hippocampus', 'Amygdala',
               'VentralDC', 'CortexVol', 'Brain-Stem', 'WMH_volume']

df_norm = normalize_volumes(imputed_df, volume_vars,
                            'EstimatedTotalIntraCranialVol')

df_norm = df_norm.drop(['EstimatedTotalIntraCranialVol'], axis=1)
# Apply
X = df_norm.drop(["subject", "Group"], axis=1)
# X = X.drop(["Left-Lateral-Ventricle", "Right-Lateral-Ventricle"],axis=1)
#X["Sex"] = X["Sex"].map({"home": 0, "dona": 1})

y = df_norm["Group"]
y = df_norm["Group"].map({"HC": 0, dataset: 1})


models = {
    "kNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}
# %% APPLICATION 5 FOLD CROSS VALIDATION
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
kfold_cv_results_exp1 = pd.DataFrame()
kfold_cv_results_exp2 = pd.DataFrame()

for model_name, model in models.items():
    results_exp1, results_exp2 = kfold_cv_classification(X, y, model_name, model, output_dir, cv=cv)
    kfold_cv_results_exp1 = pd.concat([kfold_cv_results_exp1, pd.DataFrame(results_exp1)])
    kfold_cv_results_exp2 = pd.concat([kfold_cv_results_exp2, pd.DataFrame(results_exp2)])

kfold_cv_results_exp1.to_excel(join(output_dir, "CV_results_ALLDATA.xlsx"))
kfold_cv_results_exp2.to_excel(join(output_dir, "CV_results_NIDATA.xlsx"))

# %% PLOT RESULTS
def plot_boxplots_metrics(results_df, title, output_dir, exp):
    metrics = ['accuracy', 'auc', 'specificity', 'sensitivity', 'precision']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

    
    results_long = results_df.melt(id_vars=['model', 'fold'], 
                                    value_vars=metrics,
                                    var_name='metric', 
                                    value_name='value')
    
    # Crear el plot
    plt.figure(figsize=(10, 6))
    
    # Boxplot agrupado
    sns.boxplot(data=results_long, x='model', y='value', hue='metric', 
                palette=colors)
    
    plt.title(title, fontsize=20)
    plt.xlabel('Model', fontsize=18)
    plt.ylabel('Metric', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([-0.05,1.05])
    plt.legend(title='Metric', bbox_to_anchor=(1,0.5))
    plt.savefig(join(output_dir,f"kfold_cv_metrics_folds_{exp}.jpg"),dpi=300)
    plt.show()

def metrics_mean_sd_table(results_df):
    metrics = ["accuracy", "precision", "sensitivity", "specificity", "auc"]
    
    summary = (results_df.groupby("model")[metrics].agg(["mean", "std"]))

    # Construir tabla final mean ± SD
    final_table = pd.DataFrame(index=summary.index)

    for metric in metrics:
        final_table[metric.upper()] = (
            summary[(metric, "mean")].round(3).astype(str)
            + " ± "
            + summary[(metric, "std")].round(3).astype(str)
        )

    return final_table


plot_boxplots_metrics(kfold_cv_results_exp1, f'{dataset} dataset - Metrics across folds (Neuroimaging + Demographic data)', output_dir, "EXP1")
plot_boxplots_metrics(kfold_cv_results_exp2, f'{dataset} dataset - Metrics across folds (Neuroimaging data)', output_dir, "EXP2")

summary_results_kfold = metrics_mean_sd_table(kfold_cv_results_exp1)
summary_results_kfold.to_csv(join(output_dir, "summary_metrics_folds_ALLDATA.txt"))
summary_results_kfold = metrics_mean_sd_table(kfold_cv_results_exp2)
summary_results_kfold.to_csv(join(output_dir, "summary_metrics_folds_NIDATA.txt"))
# %%
from collections import Counter

def feature_frequency_table(results_df, model_name, title, output_dir, exp):
    df_model = results_df[results_df["model"] == model_name]
    
    all_features = []
    for feats in df_model["selected_features"]:
        all_features.extend(feats)
        
    freq = Counter(all_features)
    
    freq_df = (
        pd.DataFrame.from_dict(freq, orient="index", columns=["count"])
        .sort_values("count", ascending=False)
    )
    
    freq_df["frequency_%"] = 100 * freq_df["count"] / df_model["fold"].nunique()
          
    df_plot = freq_df.rename(columns={"index": "Feature"})
    df_plot["Feature"]=df_plot.index
    
    plt.figure(figsize=(6, 4))
    
    sns.barplot(
        data=df_plot,
        x="frequency_%",
        y="Feature",
        palette=sns.color_palette("Blues_r", n_colors=len(df_plot))
    )
    
    plt.xlabel("Frequency (%)")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(join(output_dir, f"feature_selection_{exp}.jpg"), dpi=300)
    plt.show()
    
    return

feature_frequency_table(kfold_cv_results_exp1, "Random Forest",  f"{dataset} dataset - Feature selection (Neuroimaging + Demographic data)", output_dir, "EXP1")
feature_frequency_table(kfold_cv_results_exp2, "Random Forest",  f"{dataset} dataset - Feature selection (Neuroimaging data)", output_dir, "EXP2")
# %%
import pandas as pd

def hyperparameter_summary_table(results_df):
    rows = []

    for _, row in results_df.iterrows():
        model = row["model"]
        fold = row["fold"]
        params = row["best_params"]

        for param_name, param_value in params.items():
            rows.append({
                "model": model,
                "fold": fold,
                "hyperparameter": param_name,
                "value": param_value
            })

    long_df = pd.DataFrame(rows)

    summary = (
        long_df
        .groupby(["model", "hyperparameter", "value"])
        .size()
        .reset_index(name="count")
        .sort_values(["model", "hyperparameter", "count"], ascending=[True, True, False])
    )

    return summary

hyperparam_table1 = hyperparameter_summary_table(kfold_cv_results_exp1)
hyperparam_table1.to_csv(join(output_dir, "hyperparameter_tuning_ALLDATA.txt"))

hyperparam_table2 = hyperparameter_summary_table(kfold_cv_results_exp2)
hyperparam_table2.to_csv(join(output_dir, "hyperparameter_tuning_NIDATA.txt"))

# %% SIMULATION

def model_selected_features(cv_results, model_name, freq_threshold=0.5):

    df_model = cv_results[cv_results["model"] == model_name]

    # Flatten list of selected features across folds
    all_features = []
    for feat_list in df_model["selected_features"]:
        all_features.extend(feat_list)

    feature_counts = Counter(all_features)

    n_folds = df_model["fold"].nunique()
    min_freq = int(np.ceil(freq_threshold * n_folds))

    selected_feat = [feat for feat, count in feature_counts.items() if count >= min_freq]

    return selected_feat

def model_selected_params(cv_results, model_name):
   df_model = cv_results[cv_results["model"] == model_name]

   params_list = df_model["best_params"].tolist()
   params_df = pd.DataFrame(params_list)

   # Most frequent hyperparameter
   best_params = {}
   for col in params_df.columns:
       best_params[col] = params_df[col].mode()[0]

   return best_params

def generate_synthetic_dataset_TVAE(X, y, N):
    groups = y.unique()
    X_synth = []
    y_synth = []
    
    for group in groups:
        idx_group = np.where(y == group)[0]
        proportion = len(y[y==group])/len(y)
        n_group = int(round(N * proportion, 0))
        
        X_g = X.iloc[idx_group].copy()
        
        # Create metadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(X_g)
        # Update Sex as categorical
        if "Sex" in X.columns:
            metadata.update_column("Sex", sdtype='categorical')
        
        # Define TVAE
        TVAE = TVAESynthesizer(metadata,epochs=1000, batch_size=60, cuda=False) 
        # fit TVAY
        TVAE.fit(X_g)
        
        # Generate needed sample size
        synthetic_data = TVAE.sample(n_group)
        X_synth.append(synthetic_data)
        y_synth.append(pd.Series([group] * len(X_synth[-1])))
    
    X_synth_f = pd.concat(X_synth).reset_index(drop=True)
    y_synth_f = pd.concat(y_synth).reset_index(drop=True)
    
    return X_synth_f, y_synth_f


def generate_first_dataset(X, y, N):
    """Function to compute a first reduced dataset by randomly choosing indices of the original dataset"""
    groups = y.unique()
    X_sampled = []  # generate empty X and y
    y_sampled = []
    idx_sampled = []
    
    for group in groups:
        idx_group = y.index[y==group]  # obtain the indices of group in y
        proportion = len(y[y==group])/len(y)  # compute group proportion
        n_group = int(round(N * proportion,0))  # compute N by group to maintain class balance
        
        # now choose random entries of the group
        sampled_idx = np.random.choice(idx_group, size=n_group, replace=False)
        X_sampled.append(X.loc[sampled_idx])
        y_sampled.append(y.loc[sampled_idx])
        idx_sampled.extend(sampled_idx)
    
    X_sampled = pd.concat(X_sampled).reset_index(drop=True)
    y_sampled = pd.concat(y_sampled).reset_index(drop=True)
    
    return X_sampled, y_sampled, np.array(idx_sampled)


def under_sampling(X, y, X_prev, y_prev, N, indices_to_exclude):
    groups = y.unique()
    X_sampled = []  # generate empty X and y
    y_sampled = []
    idx_sampled = indices_to_exclude.tolist()
    
    new_X = X.loc[~X.index.isin(indices_to_exclude)]  # exclude already chosen indices
    new_y = y.loc[~y.index.isin(indices_to_exclude)]
    
    if N < len(X):
        for group in groups:
            # compute how many cases per group there were in first generated dataset
            n_group_prev = len(y_prev[y_prev == group])
            
            # now compute proportions
            idx_group = new_y.index[new_y == group]  # obtain the indices of group in y
            proportion = len(y[y==group])/len(y)  # compute group proportion
            n_group = int(round(N * proportion,0)) - n_group_prev  # compute N by group to maintain class balance substracting the number of N in first index
            
            # now choose random entries of the group
            sampled_idx = np.random.choice(idx_group, size=n_group, replace=False)
            X_sampled.append(new_X.loc[sampled_idx])
            y_sampled.append(new_y.loc[sampled_idx])
            idx_sampled.extend(sampled_idx)
        
        X_sampled = pd.concat(X_sampled).reset_index(drop=True)
        y_sampled = pd.concat(y_sampled).reset_index(drop=True)
        
        X_final = pd.concat([X_prev, X_sampled]).reset_index(drop=True)
        y_final = pd.concat([y_prev, y_sampled]).reset_index(drop=True)
        
        return X_final, y_final, np.array(idx_sampled)
    else:
        # N equals dataset length
        return X, y, indices_to_exclude


def over_sampling(X_synth, y_synth, X_prev, y_prev, y, N, indices_to_exclude):
    groups = y.unique()
    
    if N > len(X_prev):
        X_sampled = []  # generate empty X and y
        y_sampled = []
        idx_sampled = indices_to_exclude.tolist()
        
        new_X = X_synth.loc[~X_synth.index.isin(indices_to_exclude)]  # exclude already chosen indices
        new_y = y_synth.loc[~y_synth.index.isin(indices_to_exclude)]
        
        for group in groups:
            # compute how many cases per group there were in first generated dataset
            n_group_prev = len(y_prev[y_prev == group])
            
            # now compute proportions
            idx_group = new_y.index[new_y == group]  # obtain the indices of group in y
            proportion = len(y[y==group])/len(y)  # compute group proportion of original dataset
            n_group = int(round(N * proportion,0)) - n_group_prev  # compute N by group to maintain class balance substracting the number of N in first index
            
            # now choose random entries of the group
            sampled_idx = np.random.choice(idx_group, size=n_group, replace=False)
            X_sampled.append(new_X.loc[sampled_idx])
            y_sampled.append(new_y.loc[sampled_idx])
            idx_sampled.extend(sampled_idx)
        
        X_sampled = pd.concat(X_sampled).reset_index(drop=True)
        y_sampled = pd.concat(y_sampled).reset_index(drop=True)
        
        X_final = pd.concat([X_prev, X_sampled]).reset_index(drop=True)
        y_final = pd.concat([y_prev, y_sampled]).reset_index(drop=True)
        
        return X_final, y_final, np.array(idx_sampled)
    else:
        # N equals dataset length
        return X_prev, y_prev, indices_to_exclude


def model_application_to_n_samples_undersampling(X, y, model, model_name, cv_results, N, repetitions, n_splits=5):
    # Obtain selected features
    sel_features = model_selected_features(cv_results, model_name)
    X = X[sel_features]
    
    # Obtener mejores hiperparámetros
    best_params = model_selected_params(cv_results, model_name)
    best_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    best_model = model(**best_params)
    
    # Ajustes especiales por modelo
    if model_name == "SVM":
        best_model.probability = True
        best_model.random_state = 42
    elif model_name in ["Random Forest", "XGBoost"]:
        best_model.random_state = 42
    
    # Define 5fold cv and pipeline
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", best_model)
    ])
    
    all_results = []
    
    # do repetitions
    for rep in range(1, repetitions + 1):
        first_N = N[0]
        X_prev, y_prev, idx_prev = generate_first_dataset(X, y, first_N)
        
        # Evaluate first dataset
        y_pred = cross_val_predict(pipeline, X_prev, y_prev, cv=skf)
        y_pred_proba = cross_val_predict(pipeline, X_prev, y_prev, cv=skf, method="predict_proba")[:, 1]

        metrics, _, _, _ = evaluate_model(y_prev, y_pred, np.column_stack((1 - y_pred_proba, y_pred_proba)), model_name, do_plot=False)
        all_results.append(dict(rep=rep, n=first_N, **metrics))  # Append resulting metrics to results list
        
        for n in N[1:]:
            print(f"{model_name}: rep {rep}, n: {n}")
            
            # Perform the uner sampling
            X_new, y_new, idx_new = under_sampling(X, y, X_prev, y_prev, n, idx_prev)
            
            # Evaluate dataset
            y_pred = cross_val_predict(pipeline, X_new, y_new, cv=skf)
            y_pred_proba = cross_val_predict(pipeline, X_new, y_new, cv=skf, method="predict_proba")[:, 1]

            metrics, _, _, _ = evaluate_model(y_new, y_pred, np.column_stack((1 - y_pred_proba, y_pred_proba)), model_name, do_plot=False)
            all_results.append(dict(rep=rep, n=n, **metrics))  # append results
            
            # Update previous datasets recursively
            X_prev, y_prev, idx_prev = X_new, y_new, idx_new
        
        
    return pd.DataFrame(all_results)


def model_application_to_n_samples_oversampling(X, y, X_synth, y_synth, model, model_name, cv_results, N, repetitions, n_splits=5):
    # Obtain selected features
    sel_features = model_selected_features(cv_results, model_name)
    X = X[sel_features]
    X_synth = X_synth[sel_features]
    
    # Obtain best hyperparameters
    best_params = model_selected_params(cv_results, model_name)
    best_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    best_model = model(**best_params)
    
    # Adjust models
    if model_name == "SVM":
        best_model.probability = True
        best_model.random_state = 42
    elif model_name in ["Random Forest", "XGBoost"]:
        best_model.random_state = 42
    
    # Define kfold cv and pipeline
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", best_model)
    ])
    
    all_results = []
    
    # do repetitions
    for rep in range(1, repetitions + 1):
        first_N = N[0]
        X_prev, y_prev, idx_prev = X.copy(), y.copy(), np.array([])
        
        # Evaluate first dataset
        y_pred = cross_val_predict(pipeline, X_prev, y_prev, cv=skf)
        y_pred_proba = cross_val_predict(pipeline, X_prev, y_prev, cv=skf, method="predict_proba")[:, 1]

        metrics, _, _, _ = evaluate_model(y_prev, y_pred, np.column_stack((1 - y_pred_proba, y_pred_proba)), model_name, do_plot=False)
        all_results.append(dict(rep=rep, n=first_N, **metrics))  # Append resulting metrics to results list
        
        for n in N[1:]:
            print(f"{model_name}: rep {rep}, n: {n}")
            
            # Perform the uner sampling
            X_new, y_new, idx_new = over_sampling(X_synth, y_synth, X_prev, y_prev, y, n, idx_prev)
            
            # Evaluate dataset
            y_pred = cross_val_predict(pipeline, X_new, y_new, cv=skf)
            y_pred_proba = cross_val_predict(pipeline, X_new, y_new, cv=skf, method="predict_proba")[:, 1]

            metrics, _, _, _ = evaluate_model(y_new, y_pred, np.column_stack((1 - y_pred_proba, y_pred_proba)), model_name, do_plot=False)
            all_results.append(dict(rep=rep, n=n, **metrics))  # append results
            
            # Update previous datasets recursively
            X_prev, y_prev, idx_prev = X_new, y_new, idx_new
    
    return pd.DataFrame(all_results)


def model_application_to_n_samples(X, y, X_synth, y_synth, model, model_name, cv_results, N, repetitions, test_size=0.33):
    # split n
    N_undersampling = [v for v in N if v<len(X)]
    N_oversampling = [v for v in N if v>=len(X)]
    
    results_undersampling = model_application_to_n_samples_undersampling(X, y, model, model_name, cv_results, N_undersampling, repetitions=50)
    results_oversampling = model_application_to_n_samples_oversampling(X, y, X_synth, y_synth, model, model_name, cv_results, N_oversampling, repetitions=50)
    
    all_results = pd.concat([results_undersampling, results_oversampling])
    return pd.DataFrame(all_results)


# GENERATE LIST OF N
N = list(range(20,500,10))

# Synthetic database
X_synth, y_synth = generate_synthetic_dataset_TVAE(X, y, 500-len(X))

# compute results
results_knn = model_application_to_n_samples(X, y, X_synth, y_synth, KNeighborsClassifier, "kNN", kfold_cv_results_exp2, N, repetitions=50)
results_svm = model_application_to_n_samples(X, y, X_synth, y_synth, SVC, "SVM", kfold_cv_results_exp2, N, repetitions=50)
results_rf = model_application_to_n_samples(X, y,  X_synth, y_synth,RandomForestClassifier, "Random Forest", kfold_cv_results_exp2, N, repetitions=50)
results_xg = model_application_to_n_samples(X, y,  X_synth, y_synth,XGBClassifier, "XGBoost", kfold_cv_results_exp2, N, repetitions=50)

results_knn.to_excel(join(output_dir, "simulation_knn.xlsx"))
results_svm.to_excel(join(output_dir, "simulation_svm.xlsx"))
results_rf.to_excel(join(output_dir, "simulation_rf.xlsx"))
results_xg.to_excel(join(output_dir, "simulation_xgb.xlsx"))

# %%
def plot_results(results, model_name, dataset, output_dir):
    metrics = ['accuracy', 'roc_auc', 'specificity', 'sensitivity']
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd']

   
    stats = (results.groupby("n")[metrics].agg(["mean", "std", "var"]).reset_index())
    
    # plt.figure(figsize=(8, 5))
    # for i in range(len(metrics)):
    #     plt.plot(stats["n"], [stats[(metrics[i], "var")]], color=colors[i], label=metrics[i])
    
    # plt.xlabel("Sample size (n)")
    # plt.ylabel("Variance")
    # plt.title(f"Variance of metrics in all repetitions for {model_name}")
    # plt.legend()
    # plt.grid(alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(8, 5))
    
    for i in range(len(metrics)):
        mean = stats[(metrics[i], "mean")]
        std = stats[(metrics[i], "std")]
        
        plt.plot(stats["n"], mean, color=colors[i], label=metrics[i])
        plt.fill_between(stats["n"],mean - std, mean + std, color=colors[i], alpha=0.1)
    
    plt.title(f"{dataset} dataset - {model_name}: Performance metrics across all repetitions", fontsize=17)
    plt.xlabel("Sample size (n)")
    plt.ylabel("Metric")
    plt.xlabel('Sample size (n)', fontsize=18)
    plt.ylabel('Metric', fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim([-0.05,1.05])
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(join(output_dir, f"{model_name}_simulation.jpg"), dpi=300)
    plt.show()

plot_results(results_knn, "kNN", dataset, output_dir)
plot_results(results_svm, "SVM", dataset, output_dir)
plot_results(results_rf, "Random Forest", dataset, output_dir)
plot_results(results_xg, "XGBoost", dataset, output_dir)
