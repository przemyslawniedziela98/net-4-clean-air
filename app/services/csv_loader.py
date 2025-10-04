from __future__ import annotations
import io
import pandas as pd
from typing import Optional

class CSVLoader:
    """Load and normalize CSV files for embedding and indexing.

    Attributes:
        csv_bytes: CSV file content in bytes.
        encoding: Encoding of the CSV.
    """

    def __init__(self, csv_bytes: bytes, encoding: str = 'utf-8') -> None:
        self.csv_bytes = csv_bytes
        self.encoding = encoding

    def load(self) -> pd.DataFrame:
        """Load CSV and create a textual `document` column for embeddings.

        Returns:
            pd.DataFrame: Normalized DataFrame with 'id' and 'document' columns.
        """
        df = pd.read_csv(io.BytesIO(self.csv_bytes), encoding=self.encoding)
        df.columns = [c.strip() for c in df.columns]

        if 'Id' in df.columns:
            df.rename(columns={'Id': 'id'}, inplace=True)
        elif 'id' not in df.columns:
            df['id'] = df.index.astype(str)

        df['id'] = df['id'].apply(self._normalize_id)
        df['id'] = df['id'].fillna(pd.Series(range(len(df)))).astype(int)

        text_fields = [c for c in df.columns if c.upper() in {
            'TITLE OF THE PAPER', 'YOUR COMPLETE NAME', 'AIM OF THE PAPER',
            'MAIN FINDINGS OF THE PAPER', 'REFERENCE IN APA FORMAT',
            'TYPE OF INDOOR ENVIRONMENT', 'NOMINATE ACCORDING TO THE PAPER'
        }]

        if not text_fields:
            text_fields = [c for c, t in df.dtypes.items() if t == 'object']

        df['document'] = df.apply(lambda row: ' \n'.join(str(row[col]) for col in text_fields if pd.notnull(row[col])), axis=1)
        return df

    @staticmethod
    def _normalize_id(val) -> Optional[int]:
        if pd.isnull(val):
            return None
        try:
            return int(float(val))
        except Exception:
            return str(val)
