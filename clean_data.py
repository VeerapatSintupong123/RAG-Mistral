import pandas as pd
from keybert import KeyBERT

def convert_date(date_str):
    if not isinstance(date_str, str):
        return None
    try:
        data = date_str.replace("{'year': '", "").replace("', 'month': '", ",").replace("', 'day': '", ",").replace("'}", "")
        data = data.split(',')
        
        month_map = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }
        
        year = data[0]
        month = month_map.get(data[1], "01")
        day = f"{int(data[2]):02d}"
        return f"{year}-{month}-{day}"
    except (IndexError, ValueError, KeyError):
        return None
    
def find_keywords(row):
    if not row["keywords"] or len(row["keywords"]) == 0:
        return ','.join([kw for kw, _ in model.extract_keywords(row["abstract"])])
    return row["keywords"]

model = KeyBERT('distilbert-base-nli-mean-tokens')
column_names = ['title', 'abstract', 'classifications', 'date_publication', 'affiliation', 'references', 'keywords']

df = pd.read_csv('./scopus_data.csv', header=0, names=column_names, on_bad_lines='skip', usecols=['title', 'abstract', 'keywords', 'date_publication'])
df.drop_duplicates(subset=['title', 'abstract'], inplace=True)
df.dropna(subset=['title', 'abstract', 'date_publication'], inplace=True)

df["title"] = df["title"].str.replace("Title:", "", regex=False)
df["date_publication"] = df["date_publication"].apply(convert_date)
df.to_csv('scopus_data_cleaned.csv', index=False, header=True)

df = pd.read_csv('./scopus_data_cleaned.csv')
df["keywords"] = df.apply(find_keywords, axis=1)
df.to_csv('scopus_data_cleaned.csv', index=False, header=True, sep=';')