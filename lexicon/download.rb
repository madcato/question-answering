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

`mkdir -p data && cd data`
files.each do |file|
  `cd data && wget #{file}`
end