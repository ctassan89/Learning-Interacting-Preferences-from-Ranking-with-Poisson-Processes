# Analisi riproducibile: eseguire dalla radice del repository o da analisi_giacomo.
# Richiede R >= 4.0 e reticulate recente (con py_require).
if (!requireNamespace("reticulate", quietly = TRUE))
  stop('Installare prima reticulate: install.packages("reticulate")')
if (!reticulate::py_available(initialize = FALSE)) {
  reticulate::py_require(c("numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"))
}
candidati <- c(".", "..")
validi <- candidati[file.exists(file.path(candidati, "data/processed/features.npz")) &
                     file.exists(file.path(candidati, "data/processed/catalogue.csv"))]
if (!length(validi)) stop("Imposta la working directory sulla radice del repository con data/processed.")
repo_dir <- normalizePath(validi[1], mustWork = TRUE)
data_dir <- file.path(repo_dir, "data/processed")
np <- reticulate::import("numpy", convert = FALSE)
z <- np$load(file.path(data_dir, "features.npz"))
X <- reticulate::py_to_r(z[["X"]])
movie_ids <- as.vector(reticulate::py_to_r(z[["movieId"]]))
columns <- as.character(reticulate::py_to_r(z[["columns"]]))
z$close()
colnames(X) <- columns
catalogue <- read.csv(file.path(data_dir, "catalogue.csv"), stringsAsFactors = FALSE, check.names = FALSE)
train <- catalogue$split == "train"
if (!"genres" %in% names(catalogue)) stop("Manca genres nel catalogue condiviso.")

k_blocchi <- 32L
max_iter_nmf <- 1500L
cartella_blocchi <- file.path(repo_dir, "analisi_giacomo", "risultati_ricalcolati")

necessari <- c("X", "train", "movie_ids", "catalogue")
if (!all(vapply(necessari, exists, logical(1), inherits = TRUE))) {
  stop("Prima carica X, train, movie_ids e catalogue con il codice precedente.")
}
stopifnot(is.matrix(X), is.numeric(X), ncol(X) == 902L,
          is.logical(train), !anyNA(train), length(train) == nrow(X),
          nrow(catalogue) == nrow(X), all(is.finite(X)),
          identical(as.character(movie_ids), as.character(catalogue$movieId)))
if (!identical(unname(train), unname(catalogue$split == "train")))
  stop("train non corrisponde allo split del catalogue.")
# Verifica nomi se presenti: impedisce selezioni su colonne riordinate.
if (!is.null(colnames(X))) {
  if (!all(grepl("^genre_", colnames(X)[101:120])) ||
      !all(grepl("^keywords_code", colnames(X)[132:259])) ||
      !identical(tail(colnames(X), 3), c("year", "log_runtime", "log_budget"))) {
    stop("Ordine colonne diverso dalla matrice condivisa v2 di Sergio.")
  }
}
if (!requireNamespace("reticulate", quietly = TRUE)) install.packages("reticulate")
# Se Python e' gia' attivo, usa l'ambiente del codice precedente.
# In una sessione nuova, reticulate prepara i requisiti dichiarati.
if (!reticulate::py_available(initialize = FALSE)) {
  reticulate::py_require(c("numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"))
}

# Definizione del motore Python incorporato: non serve un file .py separato.
reticulate::py_run_string(r"---(
import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, NMF
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


def confronto_blocchi_engine(X, train, ids, genres, out_dir, k=32, max_iter=1500):
    X = np.asarray(X, dtype=np.float64)
    train = np.asarray(train, dtype=bool).ravel()
    ids = np.asarray(ids).ravel()
    genres = np.asarray(genres).ravel()
    if X.ndim != 2 or X.shape[1] != 902:
        raise ValueError('Richiesta la X condivisa v2 con 902 colonne nell ordine originale.')
    if len(train) != len(X) or len(ids) != len(X) or len(genres) != len(X):
        raise ValueError('Numero di film incoerente tra X, train, movie_ids e catalogue.')
    if not np.isfinite(X).all() or len(np.unique(ids)) != len(ids):
        raise ValueError('X non finita oppure movieId duplicati.')
    k, max_iter = int(k), int(max_iter)
    if k < 2 or k >= min(train.sum(), 248) or (~train).sum() < 2:
        raise ValueError('Servono train/heldout e 2 <= k < min(n_train,248).')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'embeddings').mkdir(exist_ok=True)
    # Indici Python, fine esclusa: esattamente la matrice v2 di Sergio.
    blocks = dict(text=(0,100), genres=(100,120), language=(120,131),
                  keywords=(131,259), cast=(259,387), directors=(387,515),
                  writers=(515,643), companies=(643,771), collection=(771,899), numeric=(899,902))
    groups = {
        'A': ['genres','keywords','text'],
        'B': ['genres','keywords','text','cast','directors','writers'],
        'C': ['genres','keywords','text','cast','directors','writers','companies','language'],
        'D': ['genres','keywords','text','cast','directors','writers','companies','language','collection','numeric']}
    metrics, spectra, thresholds, maps, embeddings = [], [], [], {}, {}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import joblib

    def quality(Y, Yhat, mean_train):
        sse = np.square(Y-Yhat).sum()
        denom = np.square(Y-mean_train).sum()
        return float(sse/Y.size), float(100*(1-sse/denom)) if denom>0 else float('nan')

    with threadpool_limits(limits=2):
        for group, names in groups.items():
            cols = np.concatenate([np.arange(*blocks[b]) for b in names])
            Y = np.ascontiguousarray(X[:,cols]); T = Y[train]
            mu = T.mean(0); total_var = T.var(0).sum()
            p = Y.shape[1]
            for method in ['PCA','SVD','NMF']:
                print(f'{group} | {method} | {p} colonne -> {k} componenti', flush=True)
                start = time.perf_counter(); converged = True; iterations = None
                if method == 'PCA':
                    model = PCA(svd_solver='full').fit(T)
                    Z = model.transform(Y)[:,:k]
                    Yhat = Z @ model.components_[:k] + model.mean_
                    pct = 100*model.explained_variance_ratio_
                    for j,v in enumerate(pct):
                        spectra.append(dict(blocco=group,metodo=method,componente=j+1,
                                            percentuale=v,cumulata=pct[:j+1].sum()))
                    cumulative = pct.cumsum()
                    for s in [50,70,80,90,95]:
                        indices = np.flatnonzero(cumulative >= s-1e-9)
                        thresholds.append(dict(blocco=group,soglia=s,componenti=int(indices[0]+1) if len(indices) else None))
                    saved = {'model':model,'k':k,'columns_zero_based':cols}
                elif method == 'SVD':
                    model = TruncatedSVD(n_components=k,n_iter=10,random_state=2026).fit(T)
                    Z = model.transform(Y); Yhat = model.inverse_transform(Z)
                    pct = 100*Z[train].var(0)/total_var
                    for j,v in enumerate(pct):
                        spectra.append(dict(blocco=group,metodo=method,componente=j+1,
                                            percentuale=v,cumulata=pct[:j+1].sum()))
                    saved = {'model':model,'columns_zero_based':cols}
                else:
                    # Nessun offset o clipping distruttivo: Y = Y_pos - Y_neg.
                    N = np.ascontiguousarray(np.concatenate([np.maximum(Y,0),np.maximum(-Y,0)],axis=1))
                    model = NMF(n_components=k, init='nndsvda',solver='cd',
                                beta_loss='frobenius',max_iter=max_iter,tol=1e-4,random_state=2026)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always',ConvergenceWarning)
                        Wtr = model.fit_transform(N[train])
                        Who = model.transform(N[~train]) # H fisso; nessun refit sui film heldout
                    converged = not any(issubclass(w.category,ConvergenceWarning) for w in caught)
                    iterations = int(model.n_iter_)
                    if not converged:
                        print('  ATTENZIONE: NMF non convergente nel limite; consultare riepilogo.',flush=True)
                    W = np.empty((len(Y),k)); W[train] = Wtr; W[~train] = Who
                    H = model.components_
                    Yhat = W @ (H[:,:p]-H[:,p:])
                    # Rimuove l'ambiguita' di scala dei fattori per salvare l'embedding.
                    norms = np.linalg.norm(H,axis=1); norms[norms == 0] = 1
                    Z = W*norms
                    saved = {'model':model,'columns_zero_based':cols,
                             'encoding':'positive_negative_split','embedding_scale':norms}
                    del N, W, Wtr, Who
                elapsed = time.perf_counter()-start
                mse_tr,r2_tr = quality(T,Yhat[train],mu)
                mse_ho,r2_ho = quality(Y[~train],Yhat[~train],mu)
                row = dict(blocco=group,metodo=method,colonne_originali=p,
                           colonne_input=2*p if method=='NMF' else p,componenti=k,
                           mse_train=mse_tr,mse_heldout=mse_ho,
                           ricostruzione_train_pct=r2_tr,ricostruzione_heldout_pct=r2_ho,
                           convergenza=converged,iterazioni_nmf=iterations,secondi=elapsed)
                metrics.append(row)
                key = f'{method}_{group}'
                embeddings[key] = Z
                # Stessa visualizzazione per tutti: PCA 2D dell'embedding kD, fit solo su train.
                projection = PCA(n_components=2,svd_solver='full').fit(Z[train])
                maps[key] = projection.transform(Z)
                saved['map_projection'] = projection
                joblib.dump(saved,out/f'{key}_modello.joblib')
                pd.DataFrame(dict(movieId=ids,**{f'dim{j+1}':Z[:,j] for j in range(k)})).to_csv(
                    out/'embeddings'/f'{key}_{k}d.csv',index=False)
                pd.DataFrame(metrics).to_csv(out/'riepilogo.csv',index=False)
                print(f'  Ricostruzione heldout: {r2_ho:.2f}% | {elapsed:.1f} s',flush=True)
                del Yhat

    scores = pd.DataFrame(metrics); spectrum = pd.DataFrame(spectra)
    spectrum.to_csv(out/'varianza_PCA_SVD.csv',index=False)
    pd.DataFrame(thresholds).to_csv(out/'soglie_PCA.csv',index=False)
    colors = dict(A='#2374AB',B='#2A9D8F',C='#E9A23B',D='#BB3E5D')
    fig, ax = plt.subplots(figsize=(9,6))
    for group in groups:
        s=spectrum[(spectrum.blocco==group)&(spectrum.metodo=='PCA')]
        ax.plot(s.componente,s.cumulata,label=group,color=colors[group])
    for y in [80,90,95]: ax.axhline(y,color='grey',lw=.7,ls=':')
    ax.set(xlabel='Numero di componenti',ylabel='Varianza cumulata sul training (%)',
           title='PCA: quattro configurazioni, denominatori diversi',ylim=(0,101))
    ax.legend(title='Blocchi');fig.tight_layout();fig.savefig(out/'01_PCA_varianza_completa.png',dpi=170);plt.close(fig)
    fig, axs = plt.subplots(2,2,figsize=(11,8))
    for group,ax in zip(groups,axs.flat):
        for method,style in [('PCA','-'),('SVD','--')]:
            s=spectrum[(spectrum.blocco==group)&(spectrum.metodo==method)&(spectrum.componente<=k)]
            ax.plot(s.componente,s.cumulata,style,label=method,lw=2)
        ax.set(title=f'Blocco {group}',xlabel='Componenti',ylabel='Varianza cumulata training (%)',ylim=(0,100))
        ax.legend()
    fig.tight_layout();fig.savefig(out/'02_PCA_SVD_prime_componenti.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6)); pos=np.arange(4)
    for j,method in enumerate(['PCA','SVD','NMF']):
        values=[float(scores[(scores.blocco==g)&(scores.metodo==method)].ricostruzione_heldout_pct.iloc[0]) for g in groups]
        bars=ax.bar(pos+(j-1)*.25,values,width=.25,label=method)
        ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
    ax.axhline(0,color='grey',lw=.7)
    ax.set(xticks=pos,xticklabels=list(groups),ylabel='100 x (1 - SSE / errore baseline media training)',
           title=f'Ricostruzione held-out con {k} componenti: confrontare entro ciascun blocco')
    ax.legend();fig.tight_layout();fig.savefig(out/'03_ricostruzione_heldout.png',dpi=170);plt.close(fig)
    first=pd.Series(genres).fillna('Sconosciuto').astype(str).str.split('|').str[0].replace('','Sconosciuto')
    top=first.value_counts().head(8).index.tolist(); labels=first.where(first.isin(top),'Altri')
    palette={g:plt.get_cmap('tab10')(j) for j,g in enumerate(top)};palette['Altri']='lightgrey'
    fig,axs=plt.subplots(3,4,figsize=(17,11))
    for row,method in enumerate(['PCA','SVD','NMF']):
        for col,group in enumerate(groups):
            ax=axs[row,col]; M=maps[f'{method}_{group}']
            for g in ['Altri']+top:
                mask=(labels==g).to_numpy()
                ax.scatter(M[mask,0],M[mask,1],s=2,alpha=.45,c=[palette[g]],rasterized=True)
            ax.set(title=f'{method} - {group}',xlabel='Proiezione PC1',ylabel='Proiezione PC2')
            ax.set_xticks([]);ax.set_yticks([])
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',linestyle='',color=palette[g],label=g) for g in top+['Altri']]
    fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=9)
    fig.suptitle(f'PCA 2D dei dodici embedding {k}D; tutti i film, colore = primo genere',fontsize=15)
    fig.tight_layout(rect=[0,.07,1,.96]);fig.savefig(out/'04_mappe_12_modelli.png',dpi=170);plt.close(fig)
    (out/'NOTE.txt').write_text('''PCA, SVD e NMF: blocchi A/B/C/D, stesso k e stesso train/heldout condiviso.
Preprocessing NON rifatto: i risultati non replicano lo split interno di Tassan.
PCA centrata; SVD non centrata; nessuna nuova standardizzazione delle colonne.
NMF applicata a [max(X,0), max(-X,0)]: trasformazione reversibile che cambia geometria
ed obiettivo di ricostruzione. I codici casuali non diventano fattori semantici positivi.
NMF ha un solo avvio NNDSVDa: risultato esplorativo, non ricerca globale ottimale.
Ricostruzione valutata nelle colonne originali (per NMF sottraendo le due parti).
Percentuale ricostruzione = 100*(1-SSE/sum((X_subset-media_train)^2)).
Puo' essere negativa; NON e' la somma di varianze di componenti NMF.
Confrontare metodi ENTRO un blocco: A e D hanno target/denominatori diversi.
Per scegliere blocchi o k aggiungere validation; non ottimizzare ripetutamente sul heldout.
Le 12 mappe sono PCA 2D di embedding kD, NON NMF diretta a due componenti;
la proiezione delle mappe e' addestrata sul training. Assi e scale non allineati tra mappe.
Gli embedding NMF sono W*norma_riga(H) per fissare l'ambiguita' di scala dei fattori.
Convergenza=False segnala un warning in fit o transform: aumentare max_iter e ricontrollare.
I .joblib contengono modelli, colonne originali (indici Python), proiezione della mappa;
per nuovi dati NMF usare lo stesso split +/- e embedding_scale dopo transform.
Gli embedding CSV sono compatibili con il formato dell'evaluatore condiviso di Sergio.
''',encoding='utf-8')
    print('\nCOMPLETATO. Risultati in: '+str(out.resolve()),flush=True)
    return scores

)---")

# Esecuzione dei 12 modelli. NMF puo' richiedere diversi minuti.
motore_blocchi <- reticulate::import_main(convert = FALSE)
risultati_blocchi <- reticulate::py_to_r(
  motore_blocchi$confronto_blocchi_engine(
    reticulate::r_to_py(X), reticulate::r_to_py(train),
    reticulate::r_to_py(as.vector(movie_ids)),
    reticulate::r_to_py(as.character(catalogue$genres)),
    cartella_blocchi, k = k_blocchi, max_iter = max_iter_nmf
  )
)
print(risultati_blocchi)
if (interactive()) View(risultati_blocchi)

# Ogni riga contiene metriche di un modello; apri i PNG nella cartella indicata.
cat("\nGrafici e risultati:", normalizePath(cartella_blocchi), "\n")
if (interactive()) {
  browseURL(normalizePath(file.path(cartella_blocchi, "04_mappe_12_modelli.png")))
}

# Esempio per caricare uno degli embedding senza rieseguire il confronto:
# embedding_PCA_A <- read.csv(file.path(cartella_blocchi, "embeddings/PCA_A_32d.csv"))
# Z_PCA_A <- as.matrix(embedding_PCA_A[, -1])
