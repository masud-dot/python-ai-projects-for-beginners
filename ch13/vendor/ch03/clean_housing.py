from common import SEED, load
from gaps import inject_gaps

df = load()
print("missing in the original:", int(df.isna().sum().sum()))

holey = inject_gaps(df, ["HouseAge", "AveRooms"], 0.02, SEED)
counts = holey.isna().sum()
print(counts[counts > 0])

print("rows before dropna:", len(holey))
print("rows after dropna: ", len(holey.dropna()))

filled = holey.fillna(holey.median())
print("missing after fillna:", int(filled.isna().sum().sum()))
