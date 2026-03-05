import pandas as pd
import preprocessing as pp

# columns
# Unnamed, id, domain, type, url, content, scraped_at, inserted_at, updated_at, title, authors, keywords, meta_keywords, meta_description, tags, summary, source,

filepath = "C:/Users/45515/OneDrive/Desktop/Studie/2_Semester_KU/GDS/Exam/995,000_rows.csv"
CHUNKSIZE = 500
preproc_articles = pd.DataFrame({'id': pd.Series([]), 'content': pd.Series([])})
print(preproc_articles)

with pd.read_csv(filepath, chunksize=CHUNKSIZE, usecols=['id', 'content'], engine="python", quotechar='"') as reader:
    i = 0
    for chunk in reader:
        i += 1
        if i < 33:
            continue
        print(i)
        preprocessed_articles = pd.DataFrame({'id': chunk['id'], 'content': pp.clean_text(chunk['content'])})
        preprocessed_articles['content'] = pp.tokenize_series(preprocessed_articles['content'])
        bad_rows = preprocessed_articles['content'][preprocessed_articles['content'].apply(lambda x: not isinstance(x, list))]
        print(bad_rows)
        print(pp.token_char_size(preprocessed_articles['content']))
        if i % 100 == 0:
            break
