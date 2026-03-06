import pandas as pd

_FILEPATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\995,000_rows.csv"
_OUTPUT_PATH = r"C:\Users\45515\OneDrive\Desktop\Studie\2_Semester_KU\GDS\Exam\Fake-News-Detection-Project\data\domain_to_types.csv"
_CHUNKSIZE = 50000
df_dict = {}


with pd.read_csv(
    _FILEPATH,
    chunksize=_CHUNKSIZE,
    usecols=['domain','type']
) as reader:
    i = 0
    for chunk in reader:
        i += 1
        print(i*len(chunk))
        chunk['domain'] = chunk['domain'].fillna('unknown')
        chunk['type'] = chunk['type'].fillna('unknown')
        for domain, type in zip(chunk['domain'], chunk['type']):
            df_dict.setdefault(domain, {'count': 0, 'types': {}})
            df_dict[domain]['count'] += 1
            if type not in df_dict[domain]['types'].keys():
                df_dict[domain]['types'][type] = 1
            else:
                df_dict[domain]['types'][type] += 1

pd.DataFrame(df_dict).T.sort_values(by=['count'], ascending=False).to_csv(_OUTPUT_PATH,mode="w",header=True)
