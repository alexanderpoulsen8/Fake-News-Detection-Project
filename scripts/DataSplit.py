import os
import duckdb

INPUT  = r"/Users/alexanderpoulsen/Downloads/995,000_rows.csv"
OUTDIR = r"/Users/alexanderpoulsen/Desktop/gds_proj/Fake-News-Detection-Project/"
SALT   = "v1"          # skift til "v2" for nyt split når vi tiløjer mere data eller prøver det størresæt
DELIM  = ","           # eller ";" er ikke lige sikker på filens struktur

TRAIN, VAL = 0.80, 0.10  # test bliver resten

os.makedirs(OUTDIR, exist_ok=True)

con = duckdb.connect()

#Læser CSV som en view 
#normalize_names=true burde gøre så mellemrum ikke fucker os hvis kollonnenavne har mellemrum
con.execute(f"""
CREATE OR REPLACE VIEW src AS
SELECT *
FROM read_csv_auto('{INPUT}',
    delim='{DELIM}',
    header=true,
    sample_size=-1,
    ignore_errors=true
);
""")

#Laver noget deterministisk "random" u baseret på id + salt
con.execute(f"""
CREATE OR REPLACE VIEW scored AS
SELECT
  *,
  (hash('{SALT}' || cast(id AS varchar))::DOUBLE / 18446744073709551616.0) AS u 
FROM src;
""") # Vi dividere med 2^64 (18446744073709551616.0) fordi hash i DuckDB retunere en unassigned int i området 0 .. 2^64-1 for at få rigtig skalering

#Exporter split (3 scans af scored-viewet)
con.execute(f"""
COPY (SELECT * EXCLUDE(u) FROM scored WHERE u < {TRAIN})
TO '{os.path.join(OUTDIR, "train.csv")}'
(HEADER, DELIMITER '{DELIM}');
""")

con.execute(f"""
COPY (SELECT * EXCLUDE(u) FROM scored WHERE u >= {TRAIN} AND u < {TRAIN + VAL})
TO '{os.path.join(OUTDIR, "val.csv")}'
(HEADER, DELIMITER '{DELIM}');
""")

con.execute(f"""
COPY (SELECT * EXCLUDE(u) FROM scored WHERE u >= {TRAIN + VAL})
TO '{os.path.join(OUTDIR, "test.csv")}'
(HEADER, DELIMITER '{DELIM}');
""")

print("Done.")