"""
*** WORD EMBEDDING NEURAL NETWORK ***

Il programma è una rete neurale minimale che genera rappresentazioni semantiche delle parole.



DESCRIZIONE

Il programma:
1) chiede all’utente di inserire una serie di frasi con cui addestrare la rete neurale;
2) estrae dalle frasi inserite le parole rilevanti, escludendo articoli, preposizioni e congiunzioni;
3) costruisce da queste parole un corpus di coppie di parole vicine semanticamente (co-occorrenze);
4) allena una rete neurale molto semplice per generare, per ogni parola, un embedding in uno spazio di 2 sole dimensioni;
5) alla fine, consente all’utente di:
    - scegliere una parola del vocabolario costruito dagli input
    - visualizzarne l’embedding in un grafico 2D
    - vedere le n (numero scelto dall'utente) parole semanticamente più vicine (secondo la rete) alla parola inserita collegate ad essa da linee.


        
CARATTERISTICHE TECNICHE

Il programma usa una rete neurale feedforward con 1 solo livello nascosto:
    input = parola
    hidden layer = vettore 2D (embedding)
    output = previsione di parole correlate

La rete viene addestrata con un processo semplificato tipo Skip-gram: a ogni coppia di parole viene chiesto di indovinare una nuova parola partendo dall’altra.

La rete impara a rappresentare ogni parola in modo che le parole correlate finiscano vicine nel piano.

Il programma è in Python puro, importa solo le librerie necessarie alla matematica della rete e alla visualizzaizone grafica dell'output, ma non si avvale di nessuna libreria specializzata in reti neurali (come TensorFlow o PyTorch).

"""



# (0) IMPORTAZIONI

import random
import math
import matplotlib.pyplot as plt



# (1) COSTRUZIONE DI UN CORPUS DI TOKEN DA INPUT

corpus = []

print("Inserisci delle frasi. Scrivi 'fine' per terminare l'inserimento.")
while True:
    riga = input("Frase > ").strip().lower()
    if riga == "fine":
        break
    parole = riga.replace(",", "").replace(".", "").replace("'", " ").split()
    for i in range(len(parole)):
        for j in range(len(parole)):
            if i != j:
                corpus.append((parole[i], parole[j]))

if len(corpus) < 10:
    print("\n🔴 Inserisci un totale di almeno 10 parole significative per generare un embedding valido.")
    exit()


# --- normalizzazione e filtri ---
# 1) vengono esclusi dai token gli articoli, le preposizioni e le congiunzioni
# 2) vengono eliminati dai token i segni di punteggiatura o gli altri segni che non sono lettere

stop_words = {
    "il", "lo", "la", "i", "gli", "le", "l", "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra", "d",
    "delle", "dalle", "dagli", "dai", "sui", "degli", "sul", "del", "dell", "dal", "dalla", "della", "dello", "al", "alla", "agli", "ai", "alle", "sull", "col",
    "e", "o", "ma", "però", "che", "anche", "non", "come", "pure", "oppure", "è", "sono", "siamo", "siete",
}

def normalize(word):
    return word.lower().strip(" '.,!?:;<>-_")


# generazione di un corpus filtrato 
filtered_corpus = []
for w1, w2 in corpus:
    n1, n2 = normalize(w1), normalize(w2)
    if n1 not in stop_words and n2 not in stop_words:
        filtered_corpus.append((n1, n2))


# controllo che il corpus di token sia sufficientemente grande per allendare la rete neurale
if len(filtered_corpus) < 10:
    print("\n🔴 Troppi stop word: non ci sono abbastanza dati significativi.")
    exit()


# costruzione del vocabolario di parole a cui sarà assegnato un embedding
vocab = sorted(set([w for pair in filtered_corpus for w in pair]))
word_to_index = {word: i for i, word in enumerate(vocab)}
index_to_word = {i: word for word, i in word_to_index.items()}
vocab_size = len(vocab)



# (2) RETE NEURALE

# parametri della rete
embedding_dim = 2
learning_rate = 0.05
epochs = 2000


# inizializzazione dei pesi
W1 = [[random.uniform(-1, 1) for _ in range(embedding_dim)] for _ in range(vocab_size)]
W2 = [[random.uniform(-1, 1) for _ in range(vocab_size)] for _ in range(embedding_dim)]


# trasformazione del vettore di numeri reali (output grezzi del modello) in una distribuzione di probabilità, 
# cioè una sequenza di valori compresi tra 0 e 1 che sommati danno 1
# esaltando i valori alti e schiacciando quelli più bassi
def softmax(x):
    max_val = max(x)
    e_x = [math.exp(i - max_val) for i in x]
    sum_e = sum(e_x)
    return [i / sum_e for i in e_x]



# (3) TRAINING DELLA RETE NERUALE
print("\nAllenamento in corso...")
for _ in range(epochs):
    x, y = random.choice(filtered_corpus)
    x_idx, y_idx = word_to_index[x], word_to_index[y]
    h = W1[x_idx]
    z = [sum(h[k] * W2[k][j] for k in range(embedding_dim)) for j in range(vocab_size)]
    y_hat = softmax(z)
    error = [0.0] * vocab_size
    error[y_idx] = 1.0
    dL_dz = [y_hat[i] - error[i] for i in range(vocab_size)]

    for i in range(embedding_dim):
        for j in range(vocab_size):
            W2[i][j] -= learning_rate * dL_dz[j] * h[i]

    for i in range(embedding_dim):
        dL_dh = sum(dL_dz[j] * W2[i][j] for j in range(vocab_size))
        W1[x_idx][i] -= learning_rate * dL_dh
print("✅ Allenamento completato!")



# (4) EMBEDDING
embeddings = {index_to_word[i]: W1[i] for i in range(vocab_size)}



# (5) VISUALIZZAZIONE PERSONALIZZATA

while True:
    print(f"Questo è il vocabolario che ho a disposizione: \n {vocab}")
    parola = input("\n🔎 Inserisci una parola del vocabolario di cui vuoi vedere l'embedding (o 'fine' per uscire): ").strip().lower()
    if parola == "fine":
        break
    parola = normalize(parola)
    if parola not in word_to_index:
        print("❗ Parola non trovata nel vocabolario.")
        continue

    try:
        n = int(input("   Quante parole vicine vuoi visualizzare? "))
    except:
        print("❗ Inserire un numero valido.")
        continue

    centro = embeddings[parola]
    distanze = []
    for w, vec in embeddings.items():
        if w != parola:
            d = math.sqrt((vec[0] - centro[0]) ** 2 + (vec[1] - centro[1]) ** 2)
            distanze.append((w, d))

    distanze.sort(key=lambda x: x[1])
    vicini = distanze[:n]


    # plot
    plt.figure(figsize=(10, 8))
    x, y = centro
    plt.scatter(x, y, color="red")
    plt.text(x + 0.01, y + 0.01, parola, fontsize=12, fontweight="bold", color="red")

    for w, _ in vicini:
        x2, y2 = embeddings[w]
        plt.scatter(x2, y2, color="blue")
        plt.text(x2 + 0.01, y2 + 0.01, w, fontsize=9)
        plt.plot([x, x2], [y, y2], linestyle="--", color="gray", linewidth=0.5)

    plt.title(f"Embedding 2D per '{parola}' e i suoi {n} vicini")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
