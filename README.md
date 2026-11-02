# Analisi Giacomo: PCA, Truncated SVD e NMF

Confronto a 32 componenti sui dati condivisi v2 di Sergio: 9.025 film, 902 colonne, 8.171 training e 854 held-out.

## Esecuzione in RStudio
1. Mettere questa cartella nella radice del repository condiviso.
2. Verificare la presenza di `data/processed/features.npz` e `data/processed/catalogue.csv` nella stessa versione v2. Non sono duplicati in questo pacchetto.
3. Impostare la working directory sulla radice del repository.
4. Installare una versione recente di reticulate: `install.packages("reticulate")`.
5. In una sessione R nuova eseguire `source("analisi_giacomo/confronto_blocchi.R")`.

Richiede R >= 4.0. Lo script R usa scikit-learn tramite reticulate: non è un'implementazione in R puro. Prima di inizializzare Python dichiara numpy, pandas, scikit-learn, matplotlib, joblib e threadpoolctl con py_require; il primo avvio può richiedere Internet. Se si usa un ambiente Python già attivo, queste dipendenze devono essere installate in quell'ambiente.
Non servono setup.R né altri script personali. I dati vengono caricati automaticamente. Il calcolo può richiedere alcuni minuti. I nuovi risultati sono in `risultati_ricalcolati/`, esclusa da Git, senza sovrascrivere quelli verificati.

## Configurazioni
| Blocco | Caratteristiche | Colonne |
|---|---|---:|
| A | generi, keywords, testo | 248 |
| B | A + attori, registi, sceneggiatori | 632 |
| C | B + produzione, lingua | 771 |
| D | C + saga, numeriche | 902 |

PCA centrata, SVD non centrata, nessuna nuova standardizzazione delle colonne: si mantengono i pesi di Sergio. Fit e proiezioni delle mappe utilizzano solo il training condiviso. Il preprocessing non viene rifatto con lo split interno degli esperimenti di Tassan: questo non ne costituisce una replica esatta.
NMF richiede input non negativi: si utilizza [max(X,0), max(-X,0)], raddoppiando le colonne. La ricostruzione originale si ottiene sottraendo le due parti. Questa trasformazione cambia l'obiettivo e la geometria; non rende semanticamente interpretabili i codici casuali.
NMF usa un avvio NNDSVDa, massimo 1500 iterazioni, tolleranza 1e-4; SVD usa seed 2026 e 10 iterazioni.

## Risultati inclusi
`risultati_verificati/` contiene quattro grafici PNG, metriche complete, varianza PCA/SVD, soglie PCA e riepilogo a 32 componenti. Le mappe sono proiezioni PCA 2D degli embedding 32D, non NMF diretta a due componenti.

Per PCA/SVD, Percentuali_32_componenti.csv riporta la somma delle varianze delle coordinate proiettate divisa per la varianza totale sul training. Per NMF riporta invece 100*(1-SSE/SST), con SST calcolata rispetto alla media del training nello spazio originale: NON è una varianza spiegata additiva. Le metriche vanno confrontate entro la stessa configurazione. Il confronto tra configurazioni usa target diversi.

I risultati inclusi sono stati calcolati con il motore Python sui file v2 del repository Tassan fornito (snapshot e1bb3c46a3d5c16de8b0fbdf822b7ea5f39f873e). Tutti i 12 modelli risultavano convergenti. L'interfaccia R non è stata eseguita nell'ambiente di preparazione, dove R non era disponibile. Versioni diverse delle librerie possono produrre piccole differenze numeriche. Le durate nella tabella dipendono dalla macchina.

I modelli .joblib e i CSV degli embedding vengono rigenerati dallo script, ma non sono inclusi per mantenere il pacchetto leggero. Per scegliere k o i blocchi usare una validation separata, senza ottimizzare ripetutamente sull'held-out.

## Caricamento su GitHub
Estrarre lo ZIP e caricare la cartella analisi_giacomo sul proprio branch. GitHub non estrae automaticamente uno ZIP caricato nel repository. Non aggiungere copie dei dati grezzi o dei dati condivisi.
