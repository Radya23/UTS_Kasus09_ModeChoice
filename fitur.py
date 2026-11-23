import pandas as pd
import numpy as np

MODES = [1, 2, 3, 4]
MODE_ATTRS = ['ttme', 'invc', 'invt', 'gc']

def audit_dan_bersihkan_long(df: pd.DataFrame) -> pd.DataFrame:
    """Audit tipe data, duplikat, missing values, dan validasi integritas target per individu."""
    df_clean = df.copy()
    
    # Hapus duplikat persis jika ada
    df_clean = df_clean.drop_duplicates()
    
    # Pastikan kolom identitas/kategori bertipe integer
    for col in ['individual', 'mode', 'choice']:
        df_clean[col] = df_clean[col].astype(int)
        
    # Validasi setiap individu hanya memiliki tepat 1 pilihan (choice == 1)
    choice_per_ind = df_clean.groupby('individual')['choice'].sum()
    valid_ids = choice_per_ind[choice_per_ind == 1].index
    df_clean = df_clean[df_clean['individual'].isin(valid_ids)].copy()
    
    return df_clean

def reshape_long_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah dataset dari format long (4 baris per individu)
    menjadi format wide (1 baris per individu).
    """
    df_clean = audit_dan_bersihkan_long(df_long)
    
    # Ekstrak target (mode dengan choice == 1)
    target_df = (
        df_clean[df_clean['choice'] == 1][['individual', 'mode']]
        .rename(columns={'mode': 'target_mode'})
    )
    
    # Ekstrak karakteristik individu (konstan di keempat baris)
    indiv_df = df_clean.groupby('individual')[['hinc', 'psize']].first().reset_index()
    
    # Pivot atribut moda menjadi kolom terpisah per moda
    pivoted = df_clean.pivot(index='individual', columns='mode', values=MODE_ATTRS)
    pivoted.columns = [f"{attr}_mode{m}" for attr, m in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    # Gabungkan menjadi satu tabel wide
    wide_df = indiv_df.merge(pivoted, on='individual').merge(target_df, on='individual')
    return wide_df

def buat_fitur_baru(X_df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat fitur rekayasa (feature engineering) tanpa kebocoran data (row-wise).
    Dipanggil setelah train-test split maupun saat inferensi di Streamlit.
    """
    X = X_df.copy()
    psize_safe = X['psize'].replace(0, 1)
    
    for m in MODES:
        # 1. Total waktu perjalanan (waktu terminal + waktu dalam kendaraan)
        X[f'total_time_mode{m}'] = X[f'ttme_mode{m}'] + X[f'invt_mode{m}']
        # 2. Biaya dalam kendaraan per orang
        X[f'invc_per_person_mode{m}'] = X[f'invc_mode{m}'] / psize_safe
        
    # 3. Selisih biaya gabungan & waktu total terhadap alternatif terbaik (termurah/tercepat)
    gc_cols = [f'gc_mode{m}' for m in MODES]
    time_cols = [f'total_time_mode{m}' for m in MODES]
    
    min_gc = X[gc_cols].min(axis=1)
    min_time = X[time_cols].min(axis=1)
    
    for m in MODES:
        X[f'gc_diff_min_mode{m}'] = X[f'gc_mode{m}'] - min_gc
        X[f'time_diff_min_mode{m}'] = X[f'total_time_mode{m}'] - min_time
        
    return X

def get_raw_feature_columns() -> list:
    """Daftar 18 kolom fitur mentah dalam format wide (sebelum rekayasa fitur)."""
    cols = ['hinc', 'psize']
    for attr in MODE_ATTRS:
        for m in MODES:
            cols.append(f"{attr}_mode{m}")
    return cols
