import pickle

with open("db.model", "rb") as f:
	model = pickle.load(f)

fs = float(input("enter fastinh sugar "))
fu = int(input("Freq urination 0 for no and 1 for yes ")
data = [[fs, fu]]
res = model.predict(data)

print("res = " , res)