import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_processing import load_and_prepare, split_train_test, CLASS_MAP

# 1. 数据加载与划分
X, y = load_and_prepare()
X_train, X_test, y_train, y_test = split_train_test(X, y)

# 2. 模型训练
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# 3. 保存模型
os.makedirs("models", exist_ok=True)
joblib.dump(clf, os.path.join("models", "rf_classifier.joblib"))
print("model is saved: models/rf_classifier.joblib")

# 4. 性能评估
y_pred = clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred,
      target_names=[k for k,v in sorted(CLASS_MAP.items(), key=lambda x:x[1])]))
print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))