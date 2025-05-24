import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt

class TrueAutoEncoder:
    """진짜 AutoEncoder 클래스 (순수 신경망 기반)"""
    
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        hidden_dim = (input_dim + encoding_dim) // 2
        
        # Encoder: input_dim → hidden_dim → encoding_dim
        self.encoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,), 
            activation='tanh',
            solver='adam',
            max_iter=2000,
            random_state=42,
            alpha=0.01
        )
        
        # Decoder: encoding_dim → hidden_dim → input_dim
        self.decoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            activation='tanh', 
            solver='adam',
            max_iter=2000,
            random_state=42,
            alpha=0.01
        )
        
    def fit(self, X, method='iterative'):
        """
        AutoEncoder 학습
        
        Parameters:
        - X: 입력 데이터
        - method: 'iterative' (번갈아 학습) 또는 'simultaneous' (동시 학습)
        """
        
        if method == 'iterative':
            # 방법 1: 번갈아가며 학습 (더 안정적)
            return self._fit_iterative(X)
        else:
            # 방법 2: 동시 학습 (더 복잡)
            return self._fit_simultaneous(X)
    
    def _fit_iterative(self, X):
        """번갈아가며 학습하는 방법"""
        print("AutoEncoder 학습 중 (Iterative method)...")
        
        # 1단계: 초기 인코딩 (랜덤)
        np.random.seed(42)
        encoded = np.random.randn(X.shape[0], self.encoding_dim) * 0.1
        
        # 반복 학습
        n_iterations = 10
        prev_mse = float('inf')
        
        for iteration in range(n_iterations):
            print(f"  Iteration {iteration + 1}/{n_iterations}")
            
            # Step 1: Decoder 학습 (encoded → X)
            self.decoder.fit(encoded, X)
            
            # Step 2: Encoder 학습 (X → encoded)
            # 목표: 현재 decoder가 잘 복원할 수 있는 encoding 찾기
            self.encoder.fit(X, encoded)
            
            # Step 3: 새로운 encoding 생성
            encoded = self.encoder.predict(X)
            
            # 재구성 오차 계산
            reconstructed = self.decoder.predict(encoded)
            mse = np.mean((X - reconstructed) ** 2)
            
            print(f"    Reconstruction MSE: {mse:.6f}")
            
            # 수렴 확인
            if abs(prev_mse - mse) < 1e-6:
                print(f"    Converged at iteration {iteration + 1}")
                break
            prev_mse = mse
        
        print(f"AutoEncoder 학습 완료. Final MSE: {mse:.6f}")
        return self
    
    def _fit_simultaneous(self, X):
        """동시 학습하는 방법 (단순화)"""
        print("AutoEncoder 학습 중 (Simultaneous method)...")
        
        # PCA로 초기 encoding 생성
        pca = PCA(n_components=self.encoding_dim, random_state=42)
        initial_encoded = pca.fit_transform(X)
        
        # 1. Encoder 학습: X → initial_encoded
        self.encoder.fit(X, initial_encoded)
        
        # 2. 실제 encoding 생성
        encoded = self.encoder.predict(X)
        
        # 3. Decoder 학습: encoded → X
        self.decoder.fit(encoded, X)
        
        # 재구성 오차 계산
        reconstructed = self.reconstruct(X)
        mse = np.mean((X - reconstructed) ** 2)
        print(f"AutoEncoder reconstruction MSE: {mse:.6f}")
        
        return self
        
    def encode(self, X):
        """데이터를 encoding 차원으로 변환"""
        return self.encoder.predict(X)
        
    def decode(self, encoded):
        """Encoding된 데이터를 원본 차원으로 복원"""
        return self.decoder.predict(encoded)
    
    def reconstruct(self, X):
        """전체 AutoEncoder 과정 (encode → decode)"""
        encoded = self.encode(X)
        return self.decode(encoded)

class SimpleAutoEncoder:
    """기존 AutoEncoder 클래스 (PCA + Decoder)"""
    
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        hidden_dim = (input_dim + encoding_dim) // 2
        
        self.decoder = MLPRegressor(
            hidden_layer_sizes=(hidden_dim,),
            activation='tanh', 
            solver='adam',
            max_iter=2000,
            random_state=42,
            alpha=0.01
        )
        
    def fit(self, X):
        # PCA로 인코딩 (신경망 사용 안함)
        pca_encoder = PCA(n_components=self.encoding_dim, random_state=42)
        encoded = pca_encoder.fit_transform(X)
        
        # Decoder만 학습: encoded → original
        self.decoder.fit(encoded, X)
        self.pca_encoder = pca_encoder
        
        # 재구성 오차 계산
        reconstructed = self.decoder.predict(encoded)
        mse = np.mean((X - reconstructed) ** 2)
        print(f"PCA+Decoder reconstruction MSE: {mse:.6f}")
        
        return self
        
    def encode(self, X):
        return self.pca_encoder.transform(X)
        
    def decode(self, encoded):
        return self.decoder.predict(encoded)
    
    def reconstruct(self, X):
        encoded = self.encode(X)
        return self.decode(encoded)

def compare_autoencoders():
    """두 AutoEncoder 비교"""
    
    # 데이터 로딩
    print("=== AutoEncoder 비교 ===")
    derm = fetch_ucirepo(name="Dermatology")
    X_raw = derm.data.features.to_numpy()
    
    # NaN 제거
    nan_mask = ~np.isnan(X_raw).any(axis=1)
    X_raw = X_raw[nan_mask]
    
    # 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    print(f"데이터 shape: {X_scaled.shape}")
    
    # 파라미터 설정
    encoding_dim = 10
    
    # 1. PCA + Decoder AutoEncoder
    print(f"\n1. PCA + Decoder AutoEncoder")
    simple_ae = SimpleAutoEncoder(input_dim=X_scaled.shape[1], encoding_dim=encoding_dim)
    simple_ae.fit(X_scaled)
    
    # 2. 진짜 AutoEncoder (번갈아 학습)
    print(f"\n2. True AutoEncoder (Iterative)")
    true_ae_iter = TrueAutoEncoder(input_dim=X_scaled.shape[1], encoding_dim=encoding_dim)
    true_ae_iter.fit(X_scaled, method='iterative')
    
    # 3. 진짜 AutoEncoder (동시 학습)
    print(f"\n3. True AutoEncoder (Simultaneous)")
    true_ae_sim = TrueAutoEncoder(input_dim=X_scaled.shape[1], encoding_dim=encoding_dim)
    true_ae_sim.fit(X_scaled, method='simultaneous')
    
    # 결과 비교
    print(f"\n=== 성능 비교 ===")
    
    # 재구성 오차
    simple_recon = simple_ae.reconstruct(X_scaled)
    true_iter_recon = true_ae_iter.reconstruct(X_scaled)
    true_sim_recon = true_ae_sim.reconstruct(X_scaled)
    
    simple_mse = np.mean((X_scaled - simple_recon) ** 2)
    true_iter_mse = np.mean((X_scaled - true_iter_recon) ** 2)
    true_sim_mse = np.mean((X_scaled - true_sim_recon) ** 2)
    
    print(f"1. PCA + Decoder MSE: {simple_mse:.6f}")
    print(f"2. True AE (Iterative) MSE: {true_iter_mse:.6f}")
    print(f"3. True AE (Simultaneous) MSE: {true_sim_mse:.6f}")
    
    # 인코딩된 특성 비교 (2D 시각화)
    simple_encoded = simple_ae.encode(X_scaled)
    true_iter_encoded = true_ae_iter.encode(X_scaled)
    true_sim_encoded = true_ae_sim.encode(X_scaled)
    
    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PCA + Decoder
    axes[0].scatter(simple_encoded[:, 0], simple_encoded[:, 1], alpha=0.7)
    axes[0].set_title(f'PCA + Decoder\nMSE: {simple_mse:.6f}')
    axes[0].set_xlabel('Encoded Dim 1')
    axes[0].set_ylabel('Encoded Dim 2')
    axes[0].grid(True, alpha=0.3)
    
    # True AE (Iterative)
    axes[1].scatter(true_iter_encoded[:, 0], true_iter_encoded[:, 1], alpha=0.7)
    axes[1].set_title(f'True AE (Iterative)\nMSE: {true_iter_mse:.6f}')
    axes[1].set_xlabel('Encoded Dim 1')
    axes[1].set_ylabel('Encoded Dim 2')
    axes[1].grid(True, alpha=0.3)
    
    # True AE (Simultaneous)
    axes[2].scatter(true_sim_encoded[:, 0], true_sim_encoded[:, 1], alpha=0.7)
    axes[2].set_title(f'True AE (Simultaneous)\nMSE: {true_sim_mse:.6f}')
    axes[2].set_xlabel('Encoded Dim 1')
    axes[2].set_ylabel('Encoded Dim 2')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('autoencoder_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return simple_ae, true_ae_iter, true_ae_sim

if __name__ == "__main__":
    simple_ae, true_ae_iter, true_ae_sim = compare_autoencoders() 