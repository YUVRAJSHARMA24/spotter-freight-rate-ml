import pandas as pd

df = pd.read_csv("data/train-test.csv")

df["route"] = df["pickup"] + " -> " + df["delivery"]

route_counts = df["route"].value_counts()

print("===== ROUTE FREQUENCY =====")

print("Total unique routes:", len(route_counts))

print("\nRoutes with only 1 load:", (route_counts == 1).sum())
print("Routes with 2 loads:", (route_counts == 2).sum())
print("Routes with 3 loads:", (route_counts == 3).sum())
print("Routes with 4 loads:", (route_counts == 4).sum())
print("Routes with 5+ loads:", (route_counts >= 5).sum())
print("Routes with 10+ loads:", (route_counts >= 10).sum())
print("Routes with 20+ loads:", (route_counts >= 20).sum())

print("\n===== ROUTE COUNT DISTRIBUTION =====")
print(route_counts.describe())