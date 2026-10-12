import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time

class Generator(nn.Module):
    def __init__(self, noise_dim, generator_dims, data_dim):
        super(Generator, self).__init__()
        layers = []
        input_dim = noise_dim
        for dim in generator_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, data_dim))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, z): return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim, discriminator_dims):
        super(Discriminator, self).__init__()
        layers = []
        input_dim = data_dim
        for dim in discriminator_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.LeakyReLU(0.2))
            input_dim = dim
        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)


class MedGAN:
    def __init__(self, noise_dim=100, generator_dims=(128, 128), discriminator_dims=(128, 128),
                 epochs=200, batch_size=100, learning_rate=1e-3, l2scale=1e-5, verbose=False):
        
        self.noise_dim = noise_dim
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.l2scale = l2scale
        self.verbose = verbose

        self.scaler = MinMaxScaler()
        self.original_columns = None
        self.processed_columns = None
        self.categorical_cols = None
        self.original_numerical_int_cols = None
        self.generator = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Utilisation du device : {self.device}")

    def _preprocess(self, df):
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        encoded_df = pd.get_dummies(df, columns=self.categorical_cols, prefix=self.categorical_cols)
        self.processed_columns = encoded_df.columns.tolist()
        processed_data = self.scaler.fit_transform(encoded_df.values.astype(np.float32))
        return np.clip(processed_data, 0, 1)

    def _postprocess(self, generated_data):
        if generated_data.shape[1] == 0: return pd.DataFrame()
        inversed_scaled_data = self.scaler.inverse_transform(generated_data)
        df_gen = pd.DataFrame(inversed_scaled_data, columns=self.processed_columns)
        df_reconstructed = pd.DataFrame()
        for col in self.original_columns:
            if col in self.categorical_cols:
                one_hot_cols = [c for c in self.processed_columns if c.startswith(f"{col}_")]
                if not one_hot_cols: continue
                reconstructed_col = df_gen[one_hot_cols].idxmax(axis=1)
                df_reconstructed[col] = reconstructed_col.apply(lambda x: x[len(col)+1:])
            else:
                df_reconstructed[col] = df_gen[col]
        if self.original_numerical_int_cols is not None:
            for col in self.original_numerical_int_cols:
                if col in df_reconstructed.columns:
                    df_reconstructed[col] = df_reconstructed[col].round().astype(int)
        return df_reconstructed[self.original_columns]
        
    def fit(self, real_data_df):
        for col in real_data_df.select_dtypes(include=np.number).columns:
            if real_data_df[col].isnull().sum() > 0: real_data_df[col].fillna(real_data_df[col].median(), inplace=True)
        for col in real_data_df.select_dtypes(include=['object', 'category']).columns:
            if real_data_df[col].isnull().sum() > 0: real_data_df[col].fillna(real_data_df[col].mode()[0], inplace=True)
            
        self.original_columns = real_data_df.columns.tolist()
        self.original_numerical_int_cols = real_data_df.select_dtypes(include=['int', 'int64']).columns.tolist()
        data = self._preprocess(real_data_df)
        self.data_dim = data.shape[1]
        if self.data_dim == 0: return
        
        self.generator = Generator(self.noise_dim, self.generator_dims, self.data_dim).to(self.device)
        self.discriminator = Discriminator(self.data_dim, self.discriminator_dims).to(self.device)
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, weight_decay=self.l2scale)
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=self.l2scale)
        criterion = nn.BCELoss()
        dataset = torch.tensor(data, dtype=torch.float32)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            for real_batch in dataloader:
                real_batch = real_batch.to(self.device)
                current_batch_size = real_batch.size(0)
                valid = torch.full((current_batch_size, 1), 0.9, device=self.device)
                fake = torch.full((current_batch_size, 1), 0.1, device=self.device)
                optimizer_G.zero_grad()
                z = torch.randn((current_batch_size, self.noise_dim)).to(self.device)
                gen_data = self.generator(z)
                g_loss = criterion(self.discriminator(gen_data), valid)
                g_loss.backward()
                optimizer_G.step()
                optimizer_D.zero_grad()
                real_loss = criterion(self.discriminator(real_batch), valid)
                fake_loss = criterion(self.discriminator(gen_data.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()

    def generate(self, n_samples=100):
        if not self.generator: raise RuntimeError("Le modèle doit être entraîné.")
        z = torch.randn(n_samples, self.noise_dim).to(self.device)
        self.generator.eval()
        with torch.no_grad():
            gen_data_processed = self.generator(z).cpu().numpy()
        self.generator.train()
        return self._postprocess(gen_data_processed)