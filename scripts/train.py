from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_model(df):
    # 🔴 CHANGE THIS AFTER SEEING YOUR COLUMNS
    target_column = df.columns[-1]  # assume last column is label

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # ⚠️ Handle non-numeric columns
    X = X.select_dtypes(include=['number'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)

    return score