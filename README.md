# WORD EMBEDDING NEURAL NETWORK

## *"You shall know a word by the company it keeps."*
### John Rupert Firth, *A synopsis of linguistic theory 1930-1955* (1957)

Il programma è una rete neurale minimale che genera rappresentazioni semantiche delle parole.
Il suo scopo è meramente didattico: esso consente di familiarizzare con la semantica distribuzionale e con l'embedding, in quanto processo "appreso". 
Il programma infatti permette all'utente di esplorare empiricamente in che modo l'input di addestramento determini l'effettivo riconoscimento dei token linguistici da parte della rete neurale e la loro immersione nello spazio della rappresentazione semantica. 



# Descrizione

Il programma:
1) chiede all’utente di inserire una serie di frasi con cui addestrare la rete neurale;
2) estrae dalle frasi inserite le parole rilevanti, escludendo articoli, preposizioni e congiunzioni;
3) costruisce da queste parole un corpus di coppie di parole vicine semanticamente (co-occorrenze);
4) allena una rete neurale molto semplice per generare, per ogni parola, un embedding in uno spazio di 2 sole dimensioni;
5) alla fine, consente all’utente di:
    - scegliere una parola del vocabolario costruito dagli input
    - visualizzarne l’embedding in un grafico 2D
    - vedere le n (numero scelto dall'utente) parole semanticamente più vicine (secondo la rete) alla parola inserita collegate ad essa da linee.


        
# Caratteristiche tecniche

Il programma usa una rete neurale feedforward con 1 solo livello nascosto:
    input = parola
    hidden layer = vettore 2D (embedding)
    output = previsione di parole correlate

La rete viene addestrata con un processo semplificato tipo Skip-gram: a ogni coppia di parole viene chiesto di indovinare una nuova parola partendo dall’altra.

La rete impara a rappresentare ogni parola in modo che le parole correlate finiscano vicine nel piano.

Il programma è in Python puro, importa solo le librerie necessarie alla matematica della rete e alla visualizzaizone grafica dell'output, ma non si avvale di nessuna libreria specializzata in reti neurali (come TensorFlow o PyTorch).
