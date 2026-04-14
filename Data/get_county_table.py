import pandas as pd
import io, requests

url = "https://www2.census.gov/geo/docs/reference/codes/files/national_county.txt"
txt = requests.get(url).text.replace('\r','')   # remove CR
# try both separators
df = pd.read_csv(io.StringIO(txt), sep='|', header=None, dtype=str)
if df.shape[1] < 5:
    df = pd.read_csv(io.StringIO(txt), sep=',', header=None, dtype=str)

# columns vary by file; typical order: state_abbr, statefp, countyfp, countyname, classfp, funcstat
df.columns = ['STATE','STATEFP','COUNTYFP','COUNTYNAME','CLASSFP','FUNCSTAT'][:df.shape[1]]
df['STATEFP'] = df['STATEFP'].astype(int).apply(lambda x: f"{x:02d}")
df['COUNTYFP'] = df['COUNTYFP'].astype(int).apply(lambda x: f"{x:03d}")
df['FIPS'] = df['STATEFP'] + df['COUNTYFP']
df['COUNTYNAME'] = df['COUNTYNAME'].str.replace(r'\s+city$', ' City', regex=True)
df['COUNTYNAME'] = df['COUNTYNAME'].str.replace(r'\s+county$', ' County', regex=True)
df['county'] = df['COUNTYNAME'] + ' ' + df['STATE']
out = df[['county','FIPS']]
out.to_csv('county_fips.csv', index=False)
print(out.head())

