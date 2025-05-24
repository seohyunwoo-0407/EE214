import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

# 데이터 로딩 및 전처리
derm = fetch_ucirepo(name="Dermatology")
X_raw = derm.data.features.to_numpy()
y_true = derm.data.targets.to_numpy().ravel()

# NaN 제거
nan_mask = ~np.isnan(X_raw).any(axis=1)
X_raw = X_raw[nan_mask]
y_true = y_true[nan_mask]

# 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# PCA 적용
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("=== PCA 후 데이터의 중요도 ===")
print(f"PCA 후 데이터 shape: {X_pca.shape}")
print(f"총 {X_pca.shape[1]}개의 주성분")

print("\n각 주성분의 분산 설명 비율:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1} (index {i}): {ratio:.4f} ({ratio*100:.2f}%)")

print(f"\n가장 중요한 주성분 2개:")
print(f"  1위: PC1 (index 0) - {pca.explained_variance_ratio_[0]*100:.2f}% 분산 설명")
print(f"  2위: PC2 (index 1) - {pca.explained_variance_ratio_[1]*100:.2f}% 분산 설명")

print("\n=== 원본 특성과 주성분의 관계 ===")
# 원본 특성의 총 기여도
feature_importance = np.sum(np.abs(pca.components_), axis=0)
top_features = np.argsort(feature_importance)[::-1][:5]

print("원본 특성별 총 기여도 (모든 주성분에 걸쳐):")
for i, feature_idx in enumerate(top_features):
    print(f"  {i+1}. 원본 Feature {feature_idx}: {feature_importance[feature_idx]:.4f}")

print("\nPC1에서 가장 기여도가 높은 원본 특성들:")
pc1_components = pca.components_[0]
pc1_abs = np.abs(pc1_components)
pc1_top_features = np.argsort(pc1_abs)[::-1][:5]

for i, feature_idx in enumerate(pc1_top_features):
    contribution = pc1_components[feature_idx]
    print(f"  {i+1}. 원본 Feature {feature_idx}: {contribution:.4f} (abs: {pc1_abs[feature_idx]:.4f})")

print("\nPC2에서 가장 기여도가 높은 원본 특성들:")
pc2_components = pca.components_[1]
pc2_abs = np.abs(pc2_components)
pc2_top_features = np.argsort(pc2_abs)[::-1][:5]

for i, feature_idx in enumerate(pc2_top_features):
    contribution = pc2_components[feature_idx]
    print(f"  {i+1}. 원본 Feature {feature_idx}: {contribution:.4f} (abs: {pc2_abs[feature_idx]:.4f})")

print("\n=== 결론 ===")
print("PCA 후 데이터에서:")
print("- index 0 = PC1 (가장 중요한 주성분)")
print("- index 1 = PC2 (두 번째로 중요한 주성분)")
print("- 이미 중요도 순으로 정렬되어 있음!") 