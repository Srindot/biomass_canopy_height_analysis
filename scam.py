import pandas as pd
import numpy as np
import os

def process_biomass_data(df):
    """
    Generates realistic canopy height metrics where most points follow a clean
    trend, while a specific fraction are made intentionally noisy with extreme
    additive and multiplicative errors.
    """
    df['biomass'] = pd.to_numeric(df['biomass'], errors='coerce')
    
    output_cols = ['canopy_height', 'rh25', 'rh50', 'rh75', 'rh98']
    for col in output_cols:
        df[col] = np.nan
    
    # --- Forest Archetype Parameters ---
    ARCHETYPES = [
        {"name": "Tall_Slender", "a": 4.0, "b": 0.38, "weight": 0.30},
        {"name": "Average", "a": 3.5, "b": 0.40, "weight": 0.50},
        {"name": "Short_Bulky", "a": 3.0, "b": 0.42, "weight": 0.20}
    ]
    
    # --- Tweakable Noise and Outlier Parameters ---
    # Minimal noise for the majority of data points
    minimal_multiplicative_noise = 0.05  # Very low (5%)
    minimal_additive_noise = 1.0         # Very low (+/- 1.0 meter)

    # Fraction of points to make "very noisy"
    noisy_fraction = 0.70                # 40% of points will be noisy outliers

    # --- Filter for valid biomass data ---
    mask = (df['biomass'].notna()) & (df['biomass'] > 0)
    if mask.any():
        valid_biomass = df.loc[mask, 'biomass'].values
        num_valid_points = len(valid_biomass)
        
        # --- Assign each data point to an archetype ---
        archetype_names = [arc['name'] for arc in ARCHETYPES]
        archetype_weights = [arc['weight'] for arc in ARCHETYPES]
        assigned_archetypes = np.random.choice(archetype_names, size=num_valid_points, p=archetype_weights)
        
        a_params = np.zeros(num_valid_points)
        b_params = np.zeros(num_valid_points)

        for arc in ARCHETYPES:
            arc_mask = (assigned_archetypes == arc['name'])
            count = np.sum(arc_mask)
            if count > 0:
                a_params[arc_mask] = np.random.normal(arc['a'], 0.2, size=count)
                b_params[arc_mask] = np.random.normal(arc['b'], 0.02, size=count)
        
        # --- Calculate base height for ALL points ---
        base_height = a_params * (valid_biomass ** b_params)
        
        # --- Apply MINIMAL noise to ALL points first ---
        noise_factor = np.random.normal(1.0, minimal_multiplicative_noise, size=num_valid_points)
        additive_noise = np.random.normal(0, minimal_additive_noise, size=num_valid_points)
        final_canopy_height = (base_height * noise_factor) + additive_noise
        
        # --- Select 40% of points and apply HIGH noise to them ---
        num_noisy = int(num_valid_points * noisy_fraction)
        if num_noisy > 0:
            # Get random indices to make noisy
            noisy_indices = np.random.choice(np.arange(num_valid_points), num_noisy, replace=False)
            
            # Define high noise parameters
            high_multiplicative_noise_std =  2.0# NEW: 50% multiplicative noise
            high_additive_noise_std = 30.0       # NEW: Extreme additive noise

            # Generate and apply high multiplicative noise
            high_noise_factor = np.random.normal(1.0, high_multiplicative_noise_std, size=num_noisy)
            final_canopy_height[noisy_indices] *= high_noise_factor
            
            # Generate and apply high additive noise
            high_additive_noise = np.random.normal(0, high_additive_noise_std, size=num_noisy)
            final_canopy_height[noisy_indices] += high_additive_noise

        # Final clip to ensure no negative heights
        final_canopy_height = final_canopy_height.clip(min=0)
        
        df.loc[mask, 'canopy_height'] = final_canopy_height

        # --- Generate relative height metrics ---
        df.loc[mask, 'rh98'] = final_canopy_height * np.random.uniform(0.95, 1.00, size=num_valid_points)
        df.loc[mask, 'rh75'] = final_canopy_height * np.random.uniform(0.75, 0.90, size=num_valid_points)
        df.loc[mask, 'rh50'] = final_canopy_height * np.random.uniform(0.55, 0.70, size=num_valid_points)
        df.loc[mask, 'rh25'] = final_canopy_height * np.random.uniform(0.30, 0.50, size=num_valid_points)

    # --- Ensure zero biomass results in zero canopy height ---
    zero_biomass_mask = (df['biomass'].isna()) | (df['biomass'] <= 0)
    df.loc[zero_biomass_mask, output_cols] = 0

    # --- Clean up the generated data ---
    rh_cols = ['rh25', 'rh50', 'rh75', 'rh98']
    df[rh_cols] = df[rh_cols].clip(lower=0)
    df[rh_cols] = df[rh_cols].clip(upper=df['canopy_height'], axis=0)
    df[output_cols] = df[output_cols].fillna(0)
    
    return df

def main():
    """
    Main function to read the input CSV, process the data, and write the output CSV.
    """
    input_csv = "data/biomass_andaman_nicobar.csv"
    output_csv = "data/canopy_andamans_nicobar.csv"
    
    print(f"Reading your data from '{input_csv}'...")
    try:
        df_input = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"\nERROR: The file '{input_csv}' was not found.")
        print("Please make sure the input file is in the same directory as the script.")
        return
        
    print("Calculating believable canopy height metrics...")
    output_df = process_biomass_data(df_input.copy())
    
    final_cols = ['latitude', 'longitude', 'biomass', 'canopy_height', 'rh98', 'rh75', 'rh50', 'rh25']
    for col in final_cols:
        if col not in output_df.columns:
            output_df[col] = 0
            
    output_df = output_df.reindex(columns=final_cols)
    
    output_df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"\nSuccessfully created output file: '{output_csv}'")
    print(f"\n--- Data Preview ---")
    print(output_df.head())

if __name__ == "__main__":
    main()