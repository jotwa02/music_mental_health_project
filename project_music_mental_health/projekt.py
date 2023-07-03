import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import VarianceThreshold
import warnings
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import resample
warnings.filterwarnings("ignore")

# załadowanie danych z Kaggla+oczyszczenie
df = pd.read_csv('mental_health.csv')

# print(df.isnull().sum())

# usunięcie zbędnych kolumn
df = df.drop(['Permissions', 'Timestamp', 'Primary streaming service', 'BPM'], axis=1)
le = LabelEncoder()
# mało wartosci nan w pozostalych cechach, wiec je usuwam
df.dropna(inplace=True)

for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col].astype(str))
    # wyswietlenie do jakich wartości liczbowych zostały przypisane str
    # print(dict(zip(le.classes_, le.transform(le.classes_))))

df_2 = pd.read_csv('ankieta.csv')

# usunięcie zbędnych kolumn
df_2 = df_2.drop(['Sygnatura czasowa', 'BPM'], axis=1)

# Define the mapping dictionaries
mapping_yes_no = {'Nie': 0, 'Tak': 1, 'nan': 2}
mapping_music = {'Muzyka Klasyczna': 0, 'Country': 1, 'EDM': 2, 'Folk': 3, 'Gospel': 4, 'Hip hop': 5, 'Jazz': 6,
                 'K pop': 7, 'Latino': 8, 'Lofi': 9, 'Metal': 10, 'Pop': 11, 'R&B': 12, 'Rap': 13, 'Rock': 14,
                 'Muzyka z gier video': 15}
maping_freq = {'nigdy': 0, 'rzadko': 1, 'czasem': 2, 'bardzo często': 3, 'często': 3}

mapping_better_worsen_no_ef = {'Poprawia': 0, 'Nie odczuwam żadnych efektów': 1, 'Pogarsza': 2}

# Convert columns to the same format as the original dataset
columns_yes_no = ['Instrumentalist', 'Composer', 'Exploratory', 'Foreign languages', 'While working']
for i in columns_yes_no:
    df_2[i].replace(mapping_yes_no, inplace=True)

# załadowanie danych z ankiety+oczyszczenie
df_2['Music effects'].replace(mapping_better_worsen_no_ef, inplace=True)
df_2['Fav genre'].replace(mapping_music, inplace=True)
frequency_music_listened = ['Frequency [Classical]',
                            'Frequency [Country]',
                            'Frequency [EDM]',
                            'Frequency [Gospel]',
                            'Frequency [Hip hop]',
                            'Frequency [Jazz]',
                            'Frequency [K pop]',
                            'Frequency [Latin]',
                            'Frequency [Lofi]',
                            'Frequency [Metal]',
                            'Frequency [Pop]',
                            'Frequency [R&B]',
                            'Frequency [Rap]',
                            'Frequency [Rock]',
                            'Frequency [Video game music]',
                            'Frequency [Folk]']
for i in frequency_music_listened:
    df_2[i].replace(maping_freq, inplace=True)

obj_to_float = ['Depression', 'Anxiety', 'Insomnia', 'OCD']
for i in obj_to_float:
    for index, value in df_2[i].items():
        if value == 'Nie':
            df_2.at[index, i] = None
        elif isinstance(value, str) and not value.isdigit():
            df_2.at[index, i] = None

for i in obj_to_float:
    df_2[i] = pd.to_numeric(df_2[i], errors='coerce')

# Drop rows with a specific value
# usuwam, bo 2 to nan
for i in columns_yes_no:
    df_2 = df_2[df_2[i] != 2]

mea = ['Age', 'Hours per day', 'Anxiety', 'Depression', 'Insomnia', 'OCD']

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
for i in mea:
    imputer1 = imputer.fit(df_2[[i]])
    df_2[i] = imputer1.transform(df_2[[i]])

df_2.dropna(inplace=True)

columns_to_change = ['Music effects', 'While working', 'Instrumentalist', 'Composer', 'Fav genre', 'Exploratory',
                     'Foreign languages', 'Frequency [Classical]',
                     'Frequency [Country]',
                     'Frequency [EDM]',
                     'Frequency [Gospel]',
                     'Frequency [Hip hop]',
                     'Frequency [Jazz]',
                     'Frequency [K pop]',
                     'Frequency [Latin]',
                     'Frequency [Lofi]',
                     'Frequency [Metal]',
                     'Frequency [Pop]',
                     'Frequency [R&B]',
                     'Frequency [Rap]',
                     'Frequency [Rock]',
                     'Frequency [Video game music]',
                     'Frequency [Folk]']

df_2[columns_to_change] = df_2[columns_to_change].astype('int32')

main_df = pd.concat([df, df_2], ignore_index=True)

# wartości cech dotyczących zdrowia psychicznego muszą przyjmować wartości od 0 do 10
m_health = ['Depression', 'Insomnia', 'Anxiety', 'OCD']
for factor in m_health:
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(main_df[[factor]])
    main_df[factor] = imputer.transform(main_df[[factor]])
    main_df[factor] = np.where((main_df[factor] < 0) | (main_df[factor] > 10), main_df[factor].mean(), main_df[factor])
    # oceny stanu psychicznego to mają być wartości całkowite
    main_df[factor] = main_df[factor].round().astype(int)

print(main_df)


def plots():
    keys = mapping_music.keys()
    genre_names = [key for key in keys]

    # Lista czynników do analizy
    factors = ['Depression', 'Insomnia', 'Anxiety', 'OCD']
    names = ['depresji', 'bezsenności', 'anxiety', 'ocd']

    for i in range(len(factors)):
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='Fav genre', y=factors[i], data=main_df)
        plt.xlabel('Rodzaj słuchanej muzyki')
        plt.ylabel(factors[i])
        plt.xticks(rotation=90)
        plt.xticks(range(len(genre_names)), genre_names, fontsize=5)
        plt.title(f'Wykres pudełkowy oceny {names[i]} w zależności od rodzaju ulubionego gatunku muzyki')

        plt.ylim(0, main_df[factors[i]].max() + 1)

    plt.xticks(range(len(genre_names)), genre_names)
    # plt.show()

    #macierz korelacji
    corr_matrix = main_df.corr()
    plt.figure(figsize=(10, 8))
    cmap = sns.color_palette("pastel")
    heatmap = sns.heatmap(corr_matrix, annot=False, cmap=cmap)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=5)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
    plt.title('Macierz korelacji')
    plt.show()

    # dystrybucja wieku
    plt.figure(figsize=(8, 6))
    sns.histplot(main_df['Age'], kde=False, bins=50)
    plt.title('Rozkład wieku')
    plt.xlabel('Wiek')
    plt.ylabel('Liczba')

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    sns.countplot(x='Anxiety', data=main_df, ax=axes[0, 0])
    axes[0, 0].set_xlabel('Lęki')
    axes[0, 0].set_ylabel('Liczba')

    sns.countplot(x='Depression', data=main_df, ax=axes[1, 0])
    axes[1, 0].set_xlabel('Depresja')
    axes[1, 0].set_ylabel('Liczba')

    sns.countplot(x='Insomnia', data=main_df, ax=axes[0, 1])
    axes[0, 1].set_xlabel('Bezsenność')
    axes[0, 1].set_ylabel('Liczba')

    sns.countplot(x='OCD', data=main_df, ax=axes[1, 1])
    axes[1, 1].set_xlabel('Zaburzenie obsesyjno-kompulsyjne')
    axes[1, 1].set_ylabel('Liczba')

    plt.suptitle("Zdrowie psychiczne respondentów")
    plt.tight_layout()


    percentage_df = main_df.groupby(['Fav genre', 'Music effects']).size().groupby(level=0).apply(
        lambda x: 100 * x / float(x.sum())).reset_index(name='Percentage')

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Fav genre', y='Percentage', hue='Music effects', data=percentage_df, dodge=False)
    plt.title('Music Effects on Mental Health by Favorite Genre')
    plt.xlabel('Favorite Genre')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45)

    # Wybór odpowiednich danych dla każdej cechy
    # Utworzenie słownika mapującego indeksy na oryginalne nazwy gatunków
    genre_mapping = {
        0: 'Klasyczna',
        1: 'Country',
        2: 'EDM',
        3: 'Folk',
        4: 'Gospel',
        5: 'Hip hop',
        6: 'Jazz',
        7: 'K pop',
        8: 'Latino',
        9: 'Lofi',
        10: 'Metal',
        11: 'Pop',
        12: 'R&B',
        13: 'Rap',
        14: 'Rock',
        15: 'gry video'}

    # słaby stan psychiczny
    depression_counts = df[df['Depression'] > 6].groupby('Fav genre').size()
    insomnia_counts = df[df['Insomnia'] > 6].groupby('Fav genre').size()
    anxiety_counts = df[df['Anxiety'] > 6].groupby('Fav genre').size()
    ocd_counts = df[df['OCD'] > 6].groupby('Fav genre').size()

    sns.set_palette("pastel")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    axes[0, 0].bar(depression_counts.index.map(genre_mapping), depression_counts.values)
    axes[0, 0].set_ylabel('Liczba osób z ciężką depresją', fontsize=12)
    axes[0, 0].set_title('Rodzaj muzyki, a ciężka depresja', fontsize=14)
    axes[0, 0].tick_params(axis='x', rotation=90, labelsize=6)

    for i, v in enumerate(depression_counts.values):
        axes[0, 0].text(i, v + 1, str(v), ha='center', va='bottom')

    axes[0, 1].bar(insomnia_counts.index.map(genre_mapping), insomnia_counts.values)
    axes[0, 1].set_ylabel('Liczba osób z bezsennością', fontsize=12)
    axes[0, 1].set_title('Rodzaj muzyki, a bezsenność', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=90, labelsize=6)

    for i, v in enumerate(insomnia_counts.values):
        axes[0, 1].text(i, v + 1, str(v), ha='center', va='bottom')

    axes[1, 0].bar(anxiety_counts.index.map(genre_mapping), anxiety_counts.values)
    axes[1, 0].set_xlabel('Rodzaj muzyki', fontsize=12)
    axes[1, 0].set_ylabel('Liczba osób z lękami', fontsize=12)
    axes[1, 0].set_title('Rodzaj muzyki, a lęki', fontsize=14)
    axes[1, 0].tick_params(axis='x', rotation=90, labelsize=6)

    for i, v in enumerate(anxiety_counts.values):
        axes[1, 0].text(i, v + 1, str(v), ha='center', va='bottom')

    axes[1, 1].bar(ocd_counts.index.map(genre_mapping), ocd_counts.values)
    axes[1, 1].set_xlabel('Rodzaj muzyki', fontsize=12)
    axes[1, 1].set_ylabel('Liczba osób z zaburzeniami obsesyjno-kompulsyjnymi', fontsize=12)
    axes[1, 1].set_title('Rodzaj muzyki, a zaburzenia obsesyjno-kompulsyjne', fontsize=14)
    axes[1, 1].tick_params(axis='x', rotation=90, labelsize=6)

    for i, v in enumerate(ocd_counts.values):
        axes[1, 1].text(i, v + 1, str(v), ha='center', va='bottom')

    plt.subplots_adjust(hspace=0.5)

    # ------------------
    # dobry stan psychiczny
    depression_counts_2 = df[df['Depression'] <= 3].groupby('Fav genre').size()
    insomnia_counts_2 = df[df['Insomnia'] <= 3].groupby('Fav genre').size()
    anxiety_counts_2 = df[df['Anxiety'] <= 3].groupby('Fav genre').size()
    ocd_counts_2 = df[df['OCD'] <= 3].groupby('Fav genre').size()

    sns.set_palette("pastel")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # Tworzenie wykresu słupkowego dla depresji
    axes[0, 0].bar(depression_counts.index.map(genre_mapping), depression_counts_2.values)
    axes[0, 0].set_ylabel('Liczba osób z brakiem depresji', fontsize=12)
    axes[0, 0].set_title('Rodzaj muzyki, a dobry stan depresji', fontsize=14)
    axes[0, 0].tick_params(axis='x', rotation=90, labelsize=6)

    for i, v in enumerate(depression_counts_2.values):
        axes[0, 0].text(i, v + 1, str(v), ha='center', va='bottom')

    axes[0, 1].bar(insomnia_counts_2.index.map(genre_mapping), insomnia_counts_2.values)
    axes[0, 1].set_ylabel('Liczba osób z brakiem bezsenności', fontsize=12)
    axes[0, 1].set_title('Rodzaj muzyki, a brak bezsenności', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=90, labelsize=6)


    for i, v in enumerate(insomnia_counts_2.values):
        axes[0, 1].text(i, v + 1, str(v), ha='center', va='bottom')


    axes[1, 0].bar(anxiety_counts_2.index.map(genre_mapping), anxiety_counts_2.values)
    axes[1, 0].set_xlabel('Rodzaj muzyki', fontsize=12)
    axes[1, 0].set_ylabel('Liczba osób z brakiem lęków', fontsize=12)
    axes[1, 0].set_title('Rodzaj muzyki, a brak lęków', fontsize=14)
    axes[1, 0].tick_params(axis='x', rotation=90, labelsize=6)


    for i, v in enumerate(anxiety_counts_2.values):
        axes[1, 0].text(i, v + 1, str(v), ha='center', va='bottom')

    axes[1, 1].bar(ocd_counts_2.index.map(genre_mapping), ocd_counts_2.values)
    axes[1, 1].set_xlabel('Rodzaj muzyki', fontsize=12)
    axes[1, 1].set_ylabel('Liczba osób bez zaburzeń obsesyjno-kompulsyjnych', fontsize=12)
    axes[1, 1].set_title('Rodzaj muzyki, a brak zaburzeń obsesyjno-kompulsyjnych', fontsize=14)
    axes[1, 1].tick_params(axis='x', rotation=90, labelsize=6)

    for i, v in enumerate(ocd_counts_2.values):
        axes[1, 1].text(i, v + 1, str(v), ha='center', va='bottom')


    plt.subplots_adjust(hspace=0.5)

    # --------------------------------------------------------------

    depression_counts_2 = df[df['Depression'] <= 3].groupby('Fav genre').size()
    insomnia_counts_2 = df[df['Insomnia'] <= 3].groupby('Fav genre').size()
    anxiety_counts_2 = df[df['Anxiety'] <= 3].groupby('Fav genre').size()
    ocd_counts_2 = df[df['OCD'] <= 3].groupby('Fav genre').size()

    sns.set_palette("pastel")

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))


    axes[0, 0].bar(depression_counts.index.map(genre_mapping), depression_counts_2.values)
    axes[0, 0].set_ylabel('Liczba osób z brakiem depresji', fontsize=12)
    axes[0, 0].set_title('Rodzaj muzyki a brak depresji', fontsize=14)
    axes[0, 0].tick_params(axis='x', rotation=90, labelsize=6)

    for i, v in enumerate(depression_counts_2.values):
        axes[0, 0].text(i, v + 1, str(v), ha='center', va='bottom')


    axes[0, 1].bar(insomnia_counts_2.index.map(genre_mapping), insomnia_counts_2.values)
    axes[0, 1].set_ylabel('Liczba osób z brakiem bezsenności', fontsize=12)
    axes[0, 1].set_title('Rodzaj muzyki, a brak bezsenności', fontsize=14)
    axes[0, 1].tick_params(axis='x', rotation=90, labelsize=6)


    for i, v in enumerate(insomnia_counts_2.values):
        axes[0, 1].text(i, v + 1, str(v), ha='center', va='bottom')

    axes[1, 0].bar(anxiety_counts_2.index.map(genre_mapping), anxiety_counts_2.values)
    axes[1, 0].set_xlabel('Rodzaj muzyki', fontsize=12)
    axes[1, 0].set_ylabel('Liczba osób z brakiem lęków', fontsize=12)
    axes[1, 0].set_title('Rodzaj muzyki, a brak lęków', fontsize=14)
    axes[1, 0].tick_params(axis='x', rotation=90, labelsize=6)


    for i, v in enumerate(anxiety_counts_2.values):
        axes[1, 0].text(i, v + 1, str(v), ha='center', va='bottom')

    axes[1, 1].bar(ocd_counts_2.index.map(genre_mapping), ocd_counts_2.values)
    axes[1, 1].set_xlabel('Rodzaj muzyki', fontsize=12)
    axes[1, 1].set_ylabel('Liczba osób bez zaburzeń obsesyjno-kompulsyjnych', fontsize=12)
    axes[1, 1].set_title('Rodzaj muzyki, a brak zaburzeń obsesyjno-kompulsyjnych', fontsize=14)
    axes[1, 1].tick_params(axis='x', rotation=90, labelsize=6)

    for i, v in enumerate(ocd_counts_2.values):
        axes[1, 1].text(i, v + 1, str(v), ha='center', va='bottom')

    plt.subplots_adjust(hspace=0.5)

    plt.show()


main_df['Mental Health Status'] = np.where(
    ((main_df['Depression'] <= 3) & (main_df['Anxiety'] <= 3) & (main_df['Insomnia'] <= 3) & (main_df['OCD'] <= 3)),
    'Dobry stan psychiczny',
    np.where(
        (main_df['Depression'] > 6) | (main_df['Anxiety'] > 6) | (main_df['Insomnia'] > 6) | (main_df['OCD'] > 6),
        'Słaba kondycja psychiczna',
        'Średnia kondycja psychiczna'
    ))

# Podział danych na podzbiory dla poszczególnych stanów psychicznych
dobry_stan_df = main_df[main_df['Mental Health Status'] == 'Dobry stan psychiczny']
srednia_kondycja_df = main_df[main_df['Mental Health Status'] == 'Średnia kondycja psychiczna']
slaba_kondycja_df = main_df[main_df['Mental Health Status'] == 'Słaba kondycja psychiczna']

x_mapping = {
    0: 'Muzyka Klasyczna',
    1: 'Country',
    2: 'EDM',
    3: 'Folk',
    4: 'Gospel',
    5: 'Hip hop',
    6: 'Jazz',
    7: 'K pop',
    8: 'Latino',
    9: 'Lofi',
    10: 'Metal',
    11: 'Pop',
    12: 'R&B',
    13: 'Rap',
    14: 'Rock',
    15: 'Muzyka z gier video'
}

def create_heatmap(df, title):

    grouped_df = df.groupby('Fav genre').size().reset_index(name='Liczba osób')

    grouped_df['Fav genre'] = grouped_df['Fav genre'].map(x_mapping)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Fav genre', y='Liczba osób', data=grouped_df, palette='YlGnBu')
    plt.title(title)
    plt.xlabel('Rodzaj muzyki')
    plt.ylabel('Liczba osób')
    plt.xticks(rotation=90)

    for index, row in grouped_df.iterrows():
        plt.text(index, row['Liczba osób'], str(row['Liczba osób']), ha='center')

    plt.show()


create_heatmap(dobry_stan_df, 'Wykres dla dobrego stanu psychicznego')
# plt.show()

create_heatmap(srednia_kondycja_df, 'Wykres dla średniej kondycji psychicznej')
# plt.show()

create_heatmap(slaba_kondycja_df, 'Wykres dla słabej kondycji psychicznej')
# plt.show()


label_mapping = {'Dobry stan psychiczny': 0, 'Średnia kondycja psychiczna': 1, 'Słaba kondycja psychiczna': 2}
main_df['Mental Health Status'] = main_df['Mental Health Status'].map(label_mapping)
variance = np.var(main_df, axis=0)
variance_df = pd.DataFrame({'Feature': main_df.columns, 'Variance': variance})

# removing features with low variance
selector = VarianceThreshold(threshold=0.9) #0.9 calkiem ok
selector.fit(main_df)
selected_features_mask = selector.get_support()
selected_data = main_df.loc[:, selected_features_mask]

features = main_df[['Age', 'Hours per day', 'Fav genre', 'Frequency [Classical]',
       'Frequency [EDM]', 'Frequency [Folk]', 'Frequency [Hip hop]',
       'Frequency [Jazz]', 'Frequency [K pop]', 'Frequency [Lofi]',
       'Frequency [Metal]', 'Frequency [R&B]', 'Frequency [Rap]',
       'Frequency [Rock]', 'Frequency [Video game music]']]


target = main_df['Mental Health Status']  # Kolumna docelowa

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# target_counts = main_df['Fav genre'].value_counts()

# próba oversamplingu; nadpróbkowanie
# W przypadku oversamplingu, zazwyczaj stosuje się go tylko do zbioru treningowego, a nie do zbioru testowego
ros = RandomOverSampler(random_state=42)

# ros=RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# skalowanie danych, bez oversamplingu
scaler1 = StandardScaler()
X_train_scaled = scaler1.fit_transform(X_train)
X_test_scaled = scaler1.transform(X_test)

# skalowanie danych z oversamplingiem
scaler2 = StandardScaler()
X_train_scaled_resampled = scaler2.fit_transform(X_train_resampled)

# ---------proba z pca
"""# Przygotowanie danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)  # X to macierz cech
y = target  # y to wektor etykiet klas

# Tworzenie obiektu PCA
pca = PCA(n_components=2)  # Ustawienie liczby głównych składowych (np. 2)

# Dopasowanie PCA do danych
X_pca = pca.fit_transform(X_scaled)

# Podział danych na zbiór treningowy i testowy
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Tworzenie obiektu klasyfikatora (np. Regresji Logistycznej)
classifier = RandomForestClassifier(class_weight='balanced')

# Dopasowanie klasyfikatora do danych
classifier.fit(X_train_pca, y_train_pca)

# Predykcja na zbiorze testowym
y_predd = classifier.predict(X_test_pca)

# Ocena wyników
accuracy = classifier.score(X_test_pca, y_test_pca)
print("Accuracy:", accuracy)
ccc=confusion_matrix(y_test_pca,y_predd)
print(ccc)"""
# ---------------------------------------------------------------

# -----wizaulizacja rodzaj słuchanej muzyki vs stan psychiczny

# Inicjalizacja walidacji krzyżowej z 5 podziałami na zbiorze treningowym

"""# Inicjalizacja modelu
model = GradientBoostingClassifier()


# Inicjalizacja walidacji krzyżowej z 5 podziałami na zbiorze treningowym
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Wykonanie walidacji krzyżowej na zbiorze treningowym i uzyskanie wyników
scores = cross_val_score(model, X_train, y_train, cv=kfold)

# Wyświetlenie wyników dla poszczególnych podziałów na zbiorze treningowym
for fold, score in enumerate(scores):
    print(f"Fold {fold+1} (zbior treningowy): {score}")

# Trenowanie modelu na całym zbiorze treningowym
model.fit(X_train, y_train)

# Ocena modelu na zbiorze testowym
test_score = model.score(X_test, y_test)
print(f"Wynik na zbiorze testowym: {test_score}")
y_test_pred = model.predict(X_test)
recall = recall_score(y_test, y_test_pred, average='weighted')
print('recall')
print(recall)
f1 = f1_score(y_test, y_test_pred, average='weighted')
print(f1)"""

# regresja logistyczna
classifier1 = LogisticRegression(solver='lbfgs', max_iter=1000)
classifier1.fit(X_train_scaled, y_train)

# regresja logistyczna+oversampling
classifier1_oversamp = LogisticRegression(solver='lbfgs', max_iter=1000)
classifier1_oversamp.fit(X_train_scaled_resampled, y_train_resampled)

# knn
classifier2 = KNeighborsClassifier(n_neighbors=7)  # Można zmienić liczbę sąsiadów
classifier2.fit(X_train_scaled, y_train)

# knn+oversampling
classifier2_oversamp = KNeighborsClassifier(n_neighbors=7)  # Można zmienić liczbę sąsiadów
classifier2_oversamp.fit(X_train_scaled_resampled, y_train_resampled)

# random forest
classifier3 = RandomForestClassifier()
classifier3.fit(X_train_scaled, y_train)

# random forest+oversampling
classifier3_oversamp = RandomForestClassifier()
classifier3_oversamp.fit(X_train_scaled_resampled, y_train_resampled)

# random forest with balanced param
classifier3_balanced = RandomForestClassifier(class_weight='balanced')
classifier3_balanced.fit(X_train_scaled, y_train)

# svm-klasyfikator wektorów nośnych
classifier4 = SVC(kernel='rbf', probability=True)  # Można dostosować rodzaj jądra SVM
classifier4.fit(X_train_scaled, y_train)

# svm+oversamp
classifier4_oversamp = SVC(kernel='rbf',probability=True)
classifier4_oversamp.fit(X_train_scaled_resampled, y_train_resampled)

# gradient boosting
classifier5 = GradientBoostingClassifier(n_estimators=100)
classifier5.fit(X_train_scaled, y_train)

# gradient boosting+oversamp
classifier5_oversamp = GradientBoostingClassifier(n_estimators=100)
classifier5_oversamp.fit(X_train_scaled_resampled, y_train_resampled)


def plot_metricss(classifiers, X_test_scaled, y_test):
    unique_classes = np.unique(y_test)

    colors = sns.color_palette("pastel")

    for class_label in unique_classes:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy_scores = []

        for classifier in classifiers:
            y_pred = classifier.predict(X_test_scaled)
            precision = precision_score(y_test, y_pred, average=None, zero_division=0)[class_label]
            recall = recall_score(y_test, y_pred, average=None)[class_label]
            f1 = f1_score(y_test, y_pred, average=None)[class_label]
            accuracy = accuracy_score(y_test, y_pred)

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)
            accuracy_scores.append(accuracy)

        fig, ax = plt.subplots()
        indices = np.arange(len(classifiers))
        width = 0.2

        classifier_labels = [
            "Logistic Regression",
            "Logistic Regression+OVERSAMPLING",
            "KNN",
            "KNN+OVERSAMPLING",
            "Random Forest",
            "Random Forest+OVERSAMPLING",
            "Random Forest+balanced",
            "SVM",
            "SVM+OVERSAMPLING",
            "Gradient Boosting",
            "Gradient Boosting+OVERSAMPLING"
        ]

        ax.bar(indices, precision_scores, width, label='Precision', color=colors[0])
        ax.bar(indices + width, recall_scores, width, label='Recall', color=colors[1])
        ax.bar(indices + 2 * width, f1_scores, width, label='F1 Score', color=colors[2])
        ax.bar(indices + 3 * width, accuracy_scores, width, label='Accuracy', color=colors[3])

        ax.set_xticks(indices + 1.5 * width)
        ax.set_xticklabels(classifier_labels, rotation='vertical')
        ax.set_ylabel("Score")
        ax.set_title(f"Metrics for Class {class_label}")
        ax.legend()
        ax.tick_params(axis='x', labelsize=9)
        for i in indices:
            ax.text(i, precision_scores[i], "{:.2f}".format(precision_scores[i]), ha='center', va='bottom', fontsize=8)
            ax.text(i + width, recall_scores[i], "{:.2f}".format(recall_scores[i]), ha='center', va='bottom',
                    fontsize=8)
            ax.text(i + 2 * width, f1_scores[i], "{:.2f}".format(f1_scores[i]), ha='center', va='bottom', fontsize=8)
            ax.text(i + 3 * width, accuracy_scores[i], "{:.2f}".format(accuracy_scores[i]), ha='center', va='bottom',
                    fontsize=8)
        plt.tight_layout()

    plt.show()


classifiers = [classifier1, classifier1_oversamp, classifier2, classifier2_oversamp, classifier3, classifier3_oversamp,
               classifier3_balanced, classifier4, classifier4_oversamp, classifier5, classifier5_oversamp]

#plot_metricss(classifiers, X_test_scaled, y_test)




def plot_metrics_bootstrap(classifiers, X_train_scaled, y_train, X_test_scaled, y_test, n_bootstrap):
    unique_classes = np.unique(y_test)
    colors = sns.color_palette("pastel")

    for class_label in unique_classes:
        precision_scores = []
        recall_scores = []
        f1_scores = []
        accuracy_scores = []

        for classifier in classifiers:
            precision_bootstrap = []
            recall_bootstrap = []
            f1_bootstrap = []
            accuracy_bootstrap = []

            for _ in range(n_bootstrap):
                X_bootstrap, y_bootstrap = resample(X_train_scaled, y_train, replace=True)
                classifier.fit(X_bootstrap, y_bootstrap)
                y_pred = classifier.predict(X_test_scaled)

                precision = precision_score(y_test, y_pred, average=None, zero_division=0)[class_label]
                recall = recall_score(y_test, y_pred, average=None)[class_label]
                f1 = f1_score(y_test, y_pred, average=None)[class_label]
                accuracy = accuracy_score(y_test, y_pred)

                precision_bootstrap.append(precision)
                recall_bootstrap.append(recall)
                f1_bootstrap.append(f1)
                accuracy_bootstrap.append(accuracy)

            precision_scores.append(np.mean(precision_bootstrap))
            recall_scores.append(np.mean(recall_bootstrap))
            f1_scores.append(np.mean(f1_bootstrap))
            accuracy_scores.append(np.mean(accuracy_bootstrap))

        fig, ax = plt.subplots()
        indices = np.arange(len(classifiers))
        width = 0.2

        classifier_labels = [
            "Logistic Regression",
            "KNN",
            "Random Forest",
            "SVM",
            "Gradient Boosting"
        ]

        ax.bar(indices, precision_scores, width, label='Precision', color=colors[0])
        ax.bar(indices + width, recall_scores, width, label='Recall', color=colors[1])
        ax.bar(indices + 2 * width, f1_scores, width, label='F1 Score', color=colors[2])
        ax.bar(indices + 3 * width, accuracy_scores, width, label='Accuracy', color=colors[3])

        ax.set_xticks(indices + 1.5 * width)
        ax.set_xticklabels(classifier_labels, rotation='vertical')
        ax.set_ylabel("Score")
        ax.set_title(f"Metrics for Class {class_label}")
        ax.legend()
        ax.tick_params(axis='x', labelsize=9)
        for i in indices:
            ax.text(i, precision_scores[i], "{:.2f}".format(precision_scores[i]), ha='center', va='bottom', fontsize=8)
            ax.text(i + width, recall_scores[i], "{:.2f}".format(recall_scores[i]), ha='center', va='bottom',
                    fontsize=8)
            ax.text(i + 2 * width, f1_scores[i], "{:.2f}".format(f1_scores[i]), ha='center', va='bottom', fontsize=8)
            ax.text(i + 3 * width, accuracy_scores[i], "{:.2f}".format(accuracy_scores[i]), ha='center', va='bottom',
                    fontsize=8)
        plt.tight_layout()

    plt.show()

classifiersss = [classifier1, classifier2, classifier3,
               classifier4, classifier5]



def plot_metricss_bootstrap(classifiers, X_train_scaled, y_train, X_test_scaled, y_test, n_bootstrap):
    metrics_labels = ['Precision', 'Recall', 'F1 Score', 'Accuracy']
    metrics_scores = []

    for classifier in classifiers:
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            X_bootstrap, y_bootstrap = resample(X_train_scaled, y_train, replace=True)
            classifier.fit(X_bootstrap, y_bootstrap)
            y_pred = classifier.predict(X_test_scaled)

            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            accuracy = accuracy_score(y_test, y_pred)

            bootstrap_scores.append((precision, recall, f1, accuracy))

        metrics_scores.append(np.mean(bootstrap_scores, axis=0))

    fig, ax = plt.subplots()
    indices = np.arange(len(classifiers))
    width = 0.2

    classifier_labels = [
        "Logistic Regression",
        "KNN",
        "Random Forest",
        "SVM",
        "Gradient Boosting"
    ]

    colors = sns.color_palette("pastel")

    for i, metric_label in enumerate(metrics_labels):
        metric_scores = [score[i] for score in metrics_scores]
        ax.bar(indices + i * width, metric_scores, width, label=metric_label, color=colors[i])

    ax.set_xticks(indices + (len(metrics_labels) / 2) * width)
    ax.set_xticklabels(classifier_labels, rotation='vertical')
    ax.set_ylabel("Score")
    ax.set_title("Metrics Comparison")
    ax.legend()
    ax.tick_params(axis='x', labelsize=9)
    for i, metric_scores in enumerate(metrics_scores):
        for j, score in enumerate(metric_scores):
            ax.text(indices[i] + j * width, score, "{:.2f}".format(score), ha='center', va='bottom', fontsize=8)
    plt.tight_layout()

    plt.show()

#plot_metricss_bootstrap(classifiersss, X_train_scaled, y_train, X_test_scaled, y_test, 40)


def plot_metrics_average(classifiers, X_test_scaled, y_test):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    accuracy_scores = []

    for classifier in classifiers:
        y_pred = classifier.predict(X_test_scaled)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    fig, ax = plt.subplots()
    indices = np.arange(len(classifiers))
    width = 0.2

    classifier_labels = [
        "Logistic Regression",
        "Logistic Regression+OVERSAMPLING",
        "KNN",
        "KNN+OVERSAMPLING",
        "Random Forest",
        "Random Forest+OVERSAMPLING",
        "Random Forest+balanced",
        "SVM",
        "SVM+OVERSAMPLING",
        "Gradient Boosting",
        "Gradient Boosting+OVERSAMPLING"
    ]

    colors = sns.color_palette("pastel")

    ax.bar(indices, precision_scores, width, label='Precision', color=colors[0])
    ax.bar(indices + width, recall_scores, width, label='Recall', color=colors[1])
    ax.bar(indices + 2 * width, f1_scores, width, label='F1 Score', color=colors[2])
    ax.bar(indices + 3 * width, accuracy_scores, width, label='Accuracy', color=colors[3])

    ax.set_xticks(indices + 1.5 * width)
    ax.set_xticklabels(classifier_labels, rotation='vertical')
    ax.set_ylabel("Wynik")
    ax.set_title("Classification metrics-average weighted")
    ax.legend()
    ax.tick_params(axis='x', labelsize=10)
    for i in indices:
        ax.text(i, precision_scores[i], "{:.2f}".format(precision_scores[i]), ha='center', va='bottom')
        ax.text(i + width, recall_scores[i], "{:.2f}".format(recall_scores[i]), ha='center', va='bottom')
        ax.text(i + 2 * width, f1_scores[i], "{:.2f}".format(f1_scores[i]), ha='center', va='bottom')
        ax.text(i + 3 * width, accuracy_scores[i], "{:.2f}".format(accuracy_scores[i]), ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


classifiers2 = [classifier1, classifier1_oversamp, classifier2, classifier2_oversamp, classifier3, classifier3_oversamp,
               classifier3_balanced]

#plot_metrics_average(classifiers, X_test_scaled, y_test)

def plot_confusion_matrix(classifiers, X_test_scaled, y_test, class_labels, classifier_titles):
    fig, axes = plt.subplots(nrows=1, ncols=len(classifiers), figsize=(5 * len(classifiers), 5))

    for ax, classifier, title in zip(axes, classifiers, classifier_titles):
        y_pred = classifier.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)

        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"Confusion Matrix-{title}", fontsize=8)
        fig.colorbar(im, ax=ax)

        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_yticks(np.arange(len(class_labels)))
        ax.set_xticklabels(class_labels, fontsize=8)
        ax.set_yticklabels(class_labels, fontsize=8)

        for i in range(len(class_labels)):
            for j in range(len(class_labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=6)

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    plt.tight_layout()
    plt.show()


class_labels = ["Dobry stan", "Średni stan", "Słaby stan"]
classifier_titles = [
                     "Gradient Boosting",
                     "Gradient Boosting+OVER"]
classifierssss = [classifier5,classifier5_oversamp]
class_labelss = ["Dobry stan", "Średni stan", "Słaby stan"]
plot_confusion_matrix(classifierssss, X_test_scaled, y_test, class_labelss, classifier_titles)


class_labels = ['Regresja logistyczna', 'Regresja logistyczna+oversamp', 'Drzewa losowe', 'Drzewa losowe+oversamp', 'Drzewa losowe balanced','SVC','SVC+oversamp','Gradient boosting','Gradient boosting+oversamp']
def zero_class_roc_curve():
    plt.figure(figsize=(8, 6))
    #klasa 0
    for classifier, label in zip(classifiers, class_labels):
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)
        y_onehot_test.shape  # (n_samples, n_classes)
        class_of_interest = 0
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

        # Obliczanie wartości FPR i TPR
        fpr, tpr, _ = roc_curve(y_onehot_test[:, class_id], y_score[:, class_id])

        # Obliczanie AUC
        roc_auc = auc(fpr, tpr)

        # Wyświetlanie krzywej ROC
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC curves:\ndobra kondycja psychiczna vs (słaba kondycja & średnia kondycja)")
    plt.legend()
    plt.show()


def first_class_roc_curve():
    plt.figure(figsize=(8, 6))  # Utwórz nowy wykres
    # klasa 1
    for classifier, label in zip(classifiers, class_labels):
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)
        y_onehot_test.shape  # (n_samples, n_classes)
        class_of_interest = 1
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

        # Obliczanie wartości FPR i TPR
        fpr, tpr, _ = roc_curve(y_onehot_test[:, class_id], y_score[:, class_id])

        # Obliczanie AUC
        roc_auc = auc(fpr, tpr)

        # Wyświetlanie krzywej ROC
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC curves:\nśrednia kondycja psychiczna vs (słaba kondycja & dobra kondycja)")
    plt.legend()
    plt.show()


def sec_class_roc_curve():
    plt.figure(figsize=(8, 6))  # Utwórz nowy wykres
    # klasa 2
    for classifier, label in zip(classifiers, class_labels):
        y_score = classifier.fit(X_train, y_train).predict_proba(X_test)
        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)
        y_onehot_test.shape  # (n_samples, n_classes)
        class_of_interest = 2
        class_id = np.flatnonzero(label_binarizer.classes_ == class_of_interest)[0]

        # Obliczanie wartości FPR i TPR
        fpr, tpr, _ = roc_curve(y_onehot_test[:, class_id], y_score[:, class_id])

        # Obliczanie AUC
        roc_auc = auc(fpr, tpr)

        # Wyświetlanie krzywej ROC
        plt.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC curves:\nsłaba kondycja psychiczna vs (dobra kondycja & średnia kondycja)")
    plt.legend()
    plt.show()
#zero_class_roc_curve()
#first_class_roc_curve()
#sec_class_roc_curve()
