
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import lime

from matplotlib import rc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    precision_recall_curve, confusion_matrix, roc_curve
)
from lime.lime_tabular import LimeTabularExplainer
from statsmodels.stats.proportion import proportion_confint
from tableone import TableOne
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.cross_decomposition import CCA


# 한글 폰트 설정
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

merged_data_path = "merged_data.csv"  
merged_data = pd.read_csv(merged_data_path)
print("Merged data loaded successfully. Shape:", merged_data.shape)

#전처리
numeric_cols = merged_data.select_dtypes(include=['float', 'int']).columns
merged_data[numeric_cols] = merged_data[numeric_cols].fillna(0)


categorical_cols = merged_data.select_dtypes(include=['object']).columns
merged_data[categorical_cols] = merged_data[categorical_cols].fillna('Unknown')



merged_data['Region'] = merged_data['Region'].fillna('Unknown')
# print(merged_data['Region'].isnull().sum())  # 처리 후 확인


# 데이터 준비
X = merged_data.drop(columns=['SGG_CD', 'SGG_CD_PREFIX', 'Region'])  
y = merged_data['Region'] 


# 범주형 데이터 인코딩
label_encoders = {}
categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# `Region` 타겟 값 변환
label_encoder = LabelEncoder()
merged_data['Region'] = label_encoder.fit_transform(merged_data['Region'])  # 숫자로 변환
regions = label_encoder.classes_  

# Unknown 값 제거
merged_data = merged_data[merged_data['Region'] != 0]  # 0은 'Unknown'에 해당하는 숫자 값



# # 숫자와 문자열 매핑 출력
# for idx, region in enumerate(label_encoder.classes_):
#     print(f"{idx}: {region}")

# # SGG_CD를 기준으로 지역명을 매핑할 딕셔너리
# region_mapping = {
#     '11': '서울특별시',
#     '26': '부산광역시',
#     '27': '대구광역시',
#     '28': '인천광역시',
#     '29': '광주광역시',
#     '30': '대전광역시',
#     '31': '울산광역시',
#     '36': '세종특별자치시',
#     '41': '경기도',
#     '42': '강원도',
#     '43': '충청북도',
#     '44': '충청남도',
#     '45': '전라북도',
#     '46': '전라남도',
#     '47': '경상북도',
#     '48': '경상남도',
#     '50': '제주특별자치도'
# }



# regions = merged_data['Region'].unique()  # 고유한 Region 값들



overall_results = {}  # 전체 결과 저장 딕셔너리

# Nested CV 결과 저장용 리스트
nested_cv_results = []


for region in regions:
    if region == 'Unknown':  # 'Unknown' 건너뛰기
        print(f"Skipping region: {region}")
        continue

    print(f"Processing binary classification for: {region}")

     # 이진 분류 대상 생성
    y_binary = (y == region).astype(int)  # 특정 region이면 1, 아니면 0

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)


    # 5. 모델 정의
    models = {
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=50, random_state=42),
        "params": {"n_estimators": [50, 100], "max_depth": [10, 20]},
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "params": {"learning_rate": [0.01, 0.1], "max_depth": [3, 6]},
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42,force_col_wise=True),
        "params": {"learning_rate": [0.01, 0.1], "max_depth": [10, 20]},
    },
    "CatBoost": {
        "model": CatBoostClassifier(verbose=0, random_state=42),
        "params": {"learning_rate": [0.01, 0.1], "depth": [6, 10]},
    },
    }
    # 6. 각 모델 성능 및 Feature Selection
    region_results = {}

    for model_name, model_info in models.items():
        print(f"Training {model_name} for region: {region}")

        model_results = {}

        model = model_info["model"]  # 모델 객체
        param_grid = model_info["params"]  # 하이퍼파라미터 그리드


        # 모델 학습
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        # 성능 지표 계산
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        # PR Curve
        if y_prob is not None:
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

        if y_prob is not None:
            # PR Curve
            plt.figure()
            plt.plot(recall_curve, precision_curve, marker='.', label=f'PR Curve ({region})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR Curve for {model_name} ({region})')
            plt.legend()
            plt.savefig(f'pr_curve_{model_name}_{region}.png')

            # ROC Curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure()
            plt.plot(fpr, tpr, marker='.', label=f'ROC Curve ({region})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve for {model_name} ({region})')
            plt.legend()
            plt.savefig(f'roc_curve_{model_name}_{region}.png')

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            features = X.columns
        elif model_name == "XGBoost":
            booster = model.get_booster()
            importances_dict = booster.get_score(importance_type="weight")
            features = list(importances_dict.keys())
            importances = np.array(list(importances_dict.values()))
        elif model_name == "CatBoost":
            importances = model.get_feature_importance()
            features = X.columns
        else:
            importances = None
            features = None

        # Feature Selection (Top 10, 30, 100)
        top_features = [10, 30, 100]
        feature_performance = {}
        for n in top_features:
            if importances is not None:
                selected_features = pd.DataFrame({'Feature': features, 'Importance': importances}).nlargest(n, 'Importance')['Feature']
                X_train_selected = X_train[selected_features]
                X_test_selected = X_test[selected_features]
                model.fit(X_train_selected, y_train)
                y_pred_selected = model.predict(X_test_selected)
                accuracy_selected = accuracy_score(y_test, y_pred_selected)
                feature_performance[f"Top {n}"] = accuracy_selected

        # T-test & Chi-square
        if hasattr(X, 'columns') and len(X.columns) > 1:
            for feature in X.columns:
                group1 = X_train[y_train == 1][feature]
                group2 = X_train[y_train == 0][feature]
                t_stat, p_value = ttest_ind(group1, group2)
                chi2, p, dof, expected = chi2_contingency(pd.crosstab(y_train, X_train[feature]))
                print(f"Feature: {feature}, T-test P-value: {p_value}, Chi2 P-value: {p}")

        model_results.update({
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "accuracy": accuracy,
                "roc_auc": roc_auc,
                "conf_matrix": conf_matrix
            },
            "feature_importance": feature_performance
        })

        region_results[model_name] = model_results

    overall_results[region] = region_results

    for model_name, model_info in models.items():
        print(f"Training {model_name} for region: {region}")
        model = model_info["model"]  # 모델 객체
        param_grid = model_info.get("params", {})  # 파라미터 그리드 확인

        # Inner CV with GridSearchCV
        if param_grid:  # 파라미터가 존재할 경우에만 GridSearchCV 사용
            grid_search = GridSearchCV(
                estimator=model, param_grid=param_grid, cv=2, scoring="accuracy", verbose=1, n_jobs=-1
            )
        else:
            print(f"No params found for {model_name}. Skipping GridSearchCV.")
            continue

        # Nested CV with reduced dataset
        try:
            scores = cross_val_score(
                grid_search,
                X.sample(frac=0.5, random_state=42),
                y.sample(frac=0.5, random_state=42),
                cv=3,
                scoring="accuracy",
                n_jobs=-1,
            )
        except Exception as e:
            print(f"Error during cross_val_score for {model_name}: {e}")
            continue

        # Nested CV 결과 저장
        nested_cv_results.append({
            "Region": region,
            "Model": model_name,
            "Mean Accuracy": np.mean(scores),
            "Std Accuracy": np.std(scores),
            "Scores": scores.tolist(),
        })
        print(f"{model_name} Nested CV Scores: {scores}")


# Nested CV 결과 저장
nested_cv_df = pd.DataFrame(nested_cv_results)
nested_cv_df.to_csv("nested_cv_results.csv", index=False)
print("Nested CV results saved.")

# TableOne
columns = ["Region", "Model", "Mean Accuracy", "Std Accuracy"]
table = TableOne(nested_cv_df, columns=columns, categorical=["Region", "Model"])
print(table)


# 8. TableOne Summary
summary_data = []
for region, model_results in overall_results.items():
    for model_name, results in model_results.items():
        metrics = results["metrics"]
        summary_data.append({
            "Region": region,
            "Model": model_name,
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1 Score": metrics["f1_score"],
            "Accuracy": metrics["accuracy"],
            "ROC AUC": metrics["roc_auc"]
        })

summary_df = pd.DataFrame(summary_data)
columns = ["Region", "Model", "Precision", "Recall", "F1 Score", "Accuracy", "ROC AUC"]
table = TableOne(summary_df, columns=columns, categorical=["Region", "Model"])
print(table)

# SHAP 분석
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary Plot
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
plt.savefig(f'shap_summary_{model_name}_{region}.png')

# Dependency Plot (Top Feature)
if hasattr(model, "feature_importances_"):
    top_feature = X.columns[np.argmax(model.feature_importances_)]
    shap.dependence_plot(top_feature, shap_values, X_test, show=False)
    plt.savefig(f'shap_dependency_{model_name}_{region}.png')

# Force Plot (First Sample)
shap.initjs()
shap.force_plot(
    explainer.expected_value[1], shap_values[1][0, :], X_test.iloc[0, :],
    matplotlib=True
).savefig(f'shap_force_{model_name}_{region}.png')

# CCA 분석
cca = CCA(n_components=2)
cca.fit(X_train, y_train.values.reshape(-1, 1))
X_train_c, y_train_c = cca.transform(X_train, y_train.values.reshape(-1, 1))

plt.figure()
plt.scatter(X_train_c[:, 0], y_train_c[:, 0], alpha=0.7)
plt.title(f'CCA Scatter Plot ({region})')
plt.xlabel('Canonical Component 1')
plt.ylabel('Canonical Component 2')
plt.savefig(f'cca_scatter_{region}.png')

# LIME 분석
explainer = LimeTabularExplainer(
    X_train.values, 
    feature_names=X.columns.tolist(), 
    class_names=['Other', region], 
    discretize_continuous=True
)

# 첫 번째 샘플 예측
sample = X_test.iloc[0].values
explanation = explainer.explain_instance(sample, model.predict_proba, num_features=10)
explanation.save_to_file(f'lime_explanation_{model_name}_{region}.html')
