from ucimlrepo import fetch_ucirepo 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

concrete_compressive_strength = fetch_ucirepo(id=165) 

x = concrete_compressive_strength.data.features 
y = concrete_compressive_strength.data.targets 

X=pd.DataFrame(x)
Y=pd.DataFrame(y)

feature_names = X.columns.tolist()

X=X.dropna()
Y=Y.dropna()
X=StandardScaler().fit_transform(X)
X=pd.DataFrame(X, columns=feature_names)

plt.figure(figsize=(16, 12))
for i in range(len(feature_names)):
    plt.subplot(2, 4, i+1)
    plt.scatter(X[feature_names[i]], Y, alpha=0.5, edgecolor='k')
    plt.title(f'{feature_names[i]} vs Concrete Compressive Strength', fontsize=8)
    plt.grid(True)

plt.tight_layout()
plt.savefig('(scaled)Concrete_Compressive_Strength_vs_Features.png')
plt.show()

