
###This graph answers, which variables have the biggest influence on layoff_risk###

importance = tree.feature_importances_

features = pd.DataFrame({
    "Feature": Pro.X.columns,
    "Importance": importance
})

features = features.sort_values(
    by="Importance",
    ascending=False
)

print(features)


plt.figure(figsize=(10,6))

plt.barh(
    features["Feature"],
    features["Importance"]
)

plt.xlabel("Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()

plt.show()









###This shows whether the forest contains shallow or deep trees###

depths = [est.get_depth() for est in tree.estimators_]

plt.hist(depths, bins=20)

plt.xlabel("Tree Depth")
plt.ylabel("Count")
plt.title("Distribution of Tree Depths")
plt.show()
