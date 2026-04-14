import pandas as pd
import numpy as np

class DemographicsLoader:
    """
    A centralized loader for EpiHiper/synthetic population demographic files.
    Standardizes column names and categorical values for downstream tools.
    """
    
    # The Single Source of Truth for column mappings
    COLUMN_MAPPINGS = {
        'admin2': 'county',
        'home_latitude': 'latitude',
        'home_longitude': 'longitude',
        'gender': 'sex'
    }

    def __init__(self, filepath, use_pyarrow=True, skiprows=1):
        self.filepath = filepath
        self.df = self._load_and_standardize(use_pyarrow, skiprows)

    def _load_and_standardize(self, use_pyarrow, skiprows):
        """Loads the CSV and applies universal schema rules."""
        engine = "pyarrow" if use_pyarrow else "c"
        try:
            df = pd.read_csv(self.filepath, engine=engine, skiprows=skiprows)
        except (ImportError, ValueError):
            print("Warning: pyarrow engine failed or not installed, falling back to default C engine.")
            df = pd.read_csv(self.filepath, skiprows=skiprows)

        # 1. Standardize Column Names (This fixes the 'county' missing issue!)
        df.rename(columns=self.COLUMN_MAPPINGS, inplace=True)

        # 2. Standardize Sex/Gender values (1 -> male, 2 -> female)
        if 'sex' in df.columns:
            # Safely map 1/2 to strings, leaving existing strings or NaNs alone
            sex_map = {1: 'male', 2: 'female', '1': 'male', '2': 'female'}
            df['sex'] = df['sex'].map(sex_map).fillna(df['sex'])

        # 3. Set Index for fast row-by-row lookups
        if 'pid' in df.columns:
            df.set_index('pid', inplace=True)

        # 4. Standardize smh_race
        if 'smh_race' in df.columns:
            smh_race_map = {'W': 'White', 'B': 'Black', 'A': 'Asian', 'L': 'Latino', 'O': 'Other'}
            df['smh_race'] = df['smh_race'].map(smh_race_map).fillna(df['smh_race'])

        # 5. Standardize hispanic to latino boolean
        if 'hispanic' in df.columns:
            df['latino'] = df['hispanic'].apply(lambda x: True if x in [2, '2'] else False)

        return df

    def get_dataframe(self):
        """
        Returns the fully standardized Pandas DataFrame.
        Best for vectorized operations (e.g., simulate_linelist.py).
        """
        return self.df

    def get_person_dict(self, pid, required_columns):
        """
        Returns a dictionary of traits for a single person, formatted as strings.
        Fills missing values with "NA".
        Best for row-by-row processing (e.g., genetic_painter.py).
        """
        res = {}
        
        try:
            # Fast index lookup
            person_series = self.df.loc[pid]
            person_exists = True
        except KeyError:
            person_series = None
            person_exists = False

        for col in required_columns:
            if col == 'pid':
                res[col] = str(pid)
            elif person_exists and col in person_series.index:
                val = person_series[col]
                res[col] = str(val) if not pd.isna(val) else "NA"
            else:
                res[col] = "NA"
                
        return res
