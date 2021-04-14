import os
import operator
import pandas as pd
import sklearn.feature_extraction as fe
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def preprocess(x):
    ls_file = list(map(lambda v: open(v, 'r').read(), x))
    return pd.Series(ls_file).replace(r'\b([A-Za-z])\1+\b', '', regex=True) \
        .replace(r'\b[A-Za-z]\b', '', regex=True)


def main():
    # Le estensioni accettate
    accepted = [".scala", ".php", ".ts", ".rs", ".java", ".html", ".ml", ".c", ".py", ".js", ".cpp", ".swift", ".cs"]
    file_num = {}
    file_list = []
    file_ext = []
    # Inizio della fase in cui cerco i file che mi interessano nei submodule di git aggiunti
    print("-------------------------------------")
    print("---- Inizio parte di indexing... ----")
    print("-------------------------------------")
    for root, dirs, files in os.walk("source-code-set"):
        path = root.split(os.sep)
        for file in files:
            name, ext = os.path.splitext(file)
            complete_name_file = "/".join(path) + "/" + file
            # Se l'estensione non è nulla
            try:
                if ext:
                    if ext in accepted:
                        add = False
                        # Controllo che non ci siano problemi con la codifica del file
                        with open(complete_name_file, encoding='utf8') as f:
                            data = f.read()
                        if ext not in file_num:
                            add = True
                            file_num[ext] = 0
                        else:
                            if file_num[ext] < 100:
                                add = True
                                file_num[ext] += 1
                        # Se il file è valido lo aggiungo
                        if add:
                            file_list.append(complete_name_file)
                            file_ext.append(ext)
            except UnicodeDecodeError:
                pass

    # Mostro in output le informazioni sui file acquisiti
    data = pd.DataFrame.from_dict(file_num, orient='index', columns=["Num"])
    data = data.sort_values(by=["Num"])
    print("Numero totale di file: {:d}".format(data.sum(axis=0).get(0)))
    print(data)
    # Disegno un grafico che le mostri
    pie_chart = data.plot.pie(y='Num', figsize=(5, 5))
    pie_chart.get_figure().savefig("language_pie_chart.pdf")
    print("-----------------------------------")
    print("---- Fine parte di indexing... ----")
    print("-----------------------------------")
    print("///////////////////////////////////")
    X_train, X_test, y_train, y_test = ms.train_test_split(file_list, file_ext, test_size=0.25, random_state=0)

    # TODO: Aggiungere i numeri
    token_pattern = r"""([0-9]+|[A-Za-z_]\w*\b|[!\#\$%\&\*\+:\-\./<=>\?@\\\^_\|\~]+|[ \t\(\),;\{\}\[\]"'`])"""
    transformer = FunctionTransformer(preprocess)
    tfidf_vectorizer = fe.text.TfidfVectorizer(ngram_range=range(1, 3), token_pattern=token_pattern, max_features=3000)

    parameters = [
        {
            'name': 'KNeighborsClassifier',
            'clf': [KNeighborsClassifier()],
            'clf__n_neighbors': list(range(1, 5)),
            'clf__weights': ["uniform", "distance"],
            'clf__n_jobs': [400]
        },
        {
            'name': 'LogisticRegression',
            'clf': [LogisticRegression()],
            'clf__max_iter': [400],
            'clf__n_jobs': [400]
        },
        {
            'name': 'RandomForestClassifier',
            'clf': [RandomForestClassifier()],
            'clf__n_jobs': [400],
            "clf__n_estimators": [200, 300, 400],
            "clf__criterion": ["gini", "entropy"],
            "clf__min_samples_split": [2, 3],
            "clf__max_features": ["sqrt", None, "log2"]
        },
        {
            'name': 'SVC',
            'clf': [SVC()],
            'clf__C': [0.001, 0.1, 1, 10, 100, 10e5],
            'clf__kernel': ['linear', 'rbf', 'sigmoid', 'precomputed'],
            'clf__gamma': ['scale', 'auto'],
            'clf__class_weight': ['balanced'],
            'clf__probability': [True]
        }
    ]

    result = []

    for parameter in parameters:
        clf = parameter['clf'][0]
        name = parameter['name']
        parameter.pop('clf')
        parameter.pop('name')

        pipe_RF = Pipeline([
            ('preprocessing', transformer),
            ('vectorizer', tfidf_vectorizer),
            ('clf', clf)]
        )
        grid = GridSearchCV(pipe_RF, param_grid=parameter, cv=3)

        grid.fit(X_train, y_train)

        # Evaluation
        print(f' - {name} Accuracy: {grid.score(X_test, y_test)}')

        # storing result
        result.append({
            'grid': grid,
            'name': name,
            'classifier': grid.best_estimator_,
            'best score': grid.best_score_,
            'best params': grid.best_params_,
            'cv': grid.cv
        })

    result = sorted(result, key=operator.itemgetter('best score'), reverse=True)

    print(
        " -> ".join([str(v['name']) + '[' + str(v['best params']) + ']' + ": " + str(v['best score']) for v in result]))
