import os
import subprocess

files = [
    "https://www.bragitoff.com/wp-content/uploads/2016/03/A.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/B.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/C.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/D.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/E.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/F.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/G.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/H.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/I.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/J.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/K.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/L.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/M.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/N.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/O.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/P.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/Q.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/R.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/S.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/T.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/U.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/V.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/W.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/X.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/Y.csv",
    "https://www.bragitoff.com/wp-content/uploads/2016/03/Z.csv"
]

os.makedirs("data", exist_ok=True)
os.chdir("data")

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

for file in files:
    subprocess.run(["wget", file])
