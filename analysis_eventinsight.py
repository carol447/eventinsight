from __future__ import annotations

import json
import re
import time
import tracemalloc
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "inputs" / "baseinicial.csv"
OUT = ROOT / "outputs" / "analise"
FIG = OUT / "figuras"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42


def load_clean() -> pd.DataFrame:
    raw = pd.read_csv(DATA)
    # O cabeçalho original está deslocado: os valores revelam a semântica real.
    df = pd.DataFrame(
        {
            "pais": raw["Festival_Name"].astype(str).str.strip(),
            "audience": pd.to_numeric(raw["Location"], errors="coerce"),
            "faixa_etaria": raw["Attendance_Numbers"].astype(str).str.strip(),
            "tipo_publico": raw["Visitor_Demographics"].astype(str).str.strip(),
            "impacto_original": raw["Economic_Impact"].astype(str).str.strip(),
            "genero": raw["Music_Genre"].astype(str).str.strip(),
        }
    )
    df["impacto_milhoes_nominais"] = df["impacto_original"].str.extract(r"([\d.,]+)")[0].str.replace(",", ".", regex=False).astype(float)
    df["moeda"] = df["impacto_original"].str.extract(r"^\s*([^\d]+)")[0].str.strip().replace({"�": "símbolo_corrompido"})
    ages = df["faixa_etaria"].str.extract(r"(\d+)\D+(\d+)").astype(float)
    df["idade_min"] = ages[0]
    df["idade_max"] = ages[1]
    df["idade_centro"] = ages.mean(axis=1)
    df["amplitude_etaria"] = ages[1] - ages[0]
    df["genero_primario"] = df["genero"].str.split("/").str[0].str.strip()
    df["porte"] = pd.cut(
        df["audience"], [-np.inf, 20_000, 80_000, 200_000, np.inf],
        labels=["Pequeno", "Médio", "Grande", "Mega"], right=True,
    )
    return df.dropna(subset=["audience", "impacto_milhoes_nominais"]).reset_index(drop=True)


def metrics(y_true, pred) -> dict:
    y_true, pred = np.asarray(y_true), np.asarray(pred)
    return {
        "MAE": mean_absolute_error(y_true, pred),
        "RMSE": mean_squared_error(y_true, pred) ** 0.5,
        "R2": r2_score(y_true, pred) if len(y_true) > 1 else np.nan,
        "MAPE_pct": np.mean(np.abs((y_true - pred) / y_true)) * 100,
        "WAPE_pct": np.sum(np.abs(y_true - pred)) / np.sum(np.abs(y_true)) * 100,
    }


def main():
    df = load_clean()
    df.to_csv(OUT / "base_limpa.csv", index=False, encoding="utf-8-sig")
    quality = {
        "linhas": len(df), "valores_unicos_audience": int(df.audience.nunique()),
        "paises_unicos": int(df.pais.nunique()), "generos_unicos": int(df.genero.nunique()),
        "generos_primarios_unicos": int(df.genero_primario.nunique()),
        "tipos_publico_unicos": int(df.tipo_publico.nunique()),
        "moedas_rotulos_unicos": int(df.moeda.nunique()),
        "duplicatas_completas_sem_rotulo": int(df.drop(columns=["impacto_original"]).duplicated().sum()),
        "correlacao_impacto_audience": float(df[["impacto_milhoes_nominais", "audience"]].corr().iloc[0, 1]),
        "contagem_por_porte": df.porte.value_counts().to_dict(),
    }
    (OUT / "qualidade_dados.json").write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding="utf-8")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df.audience, bins=20, ax=ax)
    ax.set(title="Distribuição do público", xlabel="Público", ylabel="Eventos")
    fig.tight_layout(); fig.savefig(FIG / "distribuicao_publico.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="impacto_milhoes_nominais", y="audience", hue="porte", ax=ax)
    ax.set(title="Impacto nominal vs. público (moedas não comparáveis)", xlabel="Impacto (milhões, moeda original)", ylabel="Público")
    fig.tight_layout(); fig.savefig(FIG / "impacto_publico.png", dpi=180); plt.close(fig)

    feature_sets = {
        "M1_impacto": (["impacto_milhoes_nominais"], []),
        "M2_impacto_demografia": (["impacto_milhoes_nominais", "idade_centro", "amplitude_etaria"], []),
        "M3_impacto_genero": (["impacto_milhoes_nominais", "idade_centro"], ["genero_primario"]),
        "M4_todas_sem_alta_cardinalidade": (["impacto_milhoes_nominais", "idade_centro", "amplitude_etaria"], ["genero_primario"]),
    }
    models = {
        "Regressão Linear": LinearRegression(),
        "Árvore de Decisão": DecisionTreeRegressor(random_state=SEED, min_samples_leaf=3),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, min_samples_leaf=2, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    comparisons = []
    for fs_name, (nums, cats) in feature_sets.items():
        pre = ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), nums),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=3), cats),
        ])
        for model_name, estimator in models.items():
            pipe = Pipeline([("pre", pre), ("model", estimator)])
            start = time.perf_counter()
            scores = cross_validate(pipe, df[nums + cats], np.log1p(df.audience), cv=cv,
                                    scoring=("neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"), n_jobs=-1)
            elapsed = time.perf_counter() - start
            # Métricas CV acima são na escala log; predições OOF explícitas fornecem escala real.
            preds = np.zeros(len(df))
            for train, test in cv.split(df):
                pipe.fit(df.loc[train, nums + cats], np.log1p(df.loc[train, "audience"]))
                preds[test] = np.expm1(pipe.predict(df.loc[test, nums + cats]))
            row = {"features": fs_name, "modelo": model_name, **metrics(df.audience, preds),
                   "R2_log_medio": scores["test_r2"].mean(), "R2_log_dp": scores["test_r2"].std(),
                   "tempo_cv_s": elapsed}
            comparisons.append(row)
    comp = pd.DataFrame(comparisons).sort_values(["MAE", "RMSE"])
    comp.to_csv(OUT / "comparacao_modelos_cv.csv", index=False, encoding="utf-8-sig")

    # Seleção: importância por permutação e RFE apenas em variáveis numéricas, evitando milhares de dummies instáveis.
    numeric = ["impacto_milhoes_nominais", "idade_min", "idade_max", "idade_centro", "amplitude_etaria"]
    Xn, ylog = df[numeric], np.log1p(df.audience)
    rf = RandomForestRegressor(n_estimators=500, random_state=SEED, min_samples_leaf=2, n_jobs=-1).fit(Xn, ylog)
    pi = permutation_importance(rf, Xn, ylog, n_repeats=30, random_state=SEED, scoring="neg_mean_absolute_error")
    importance = pd.DataFrame({"atributo": numeric, "importancia_permutacao": pi.importances_mean,
                               "desvio": pi.importances_std}).sort_values("importancia_permutacao", ascending=False)
    importance.to_csv(OUT / "feature_importance.csv", index=False, encoding="utf-8-sig")
    rfe_rows = []
    for n in range(1, len(numeric) + 1):
        selector = RFE(LinearRegression(), n_features_to_select=n).fit(StandardScaler().fit_transform(Xn), ylog)
        rfe_rows.append({"n_atributos": n, "selecionados": ", ".join(np.array(numeric)[selector.support_])})
    pd.DataFrame(rfe_rows).to_csv(OUT / "rfe_selecao.csv", index=False, encoding="utf-8-sig")

    # Fine tuning no conjunto parcimonioso vencedor a priori.
    X = df[["impacto_milhoes_nominais", "idade_centro", "amplitude_etaria"]]
    grid = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1), {
        "n_estimators": [200, 500], "max_depth": [None, 4, 8],
        "min_samples_leaf": [1, 2, 4], "min_samples_split": [2, 5, 10],
        "max_features": [1.0, "sqrt"],
    }, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1)
    grid.fit(X, ylog)
    (OUT / "fine_tuning.json").write_text(json.dumps({"melhores_parametros": grid.best_params_,
        "melhor_MAE_log_cv": -grid.best_score_}, indent=2, ensure_ascii=False), encoding="utf-8")

    # Holdout estratificado por porte para análise de treino, teste e erros por porte.
    train, test = train_test_split(np.arange(len(df)), test_size=.25, random_state=SEED, stratify=df.porte)
    tuned = grid.best_estimator_.fit(X.iloc[train], ylog.iloc[train])
    pred_train = np.expm1(tuned.predict(X.iloc[train])); pred_test = np.expm1(tuned.predict(X.iloc[test]))
    overfit = {"treino": metrics(df.audience.iloc[train], pred_train), "teste": metrics(df.audience.iloc[test], pred_test)}
    (OUT / "overfitting_holdout.json").write_text(json.dumps(overfit, indent=2, ensure_ascii=False), encoding="utf-8")
    err = df.iloc[test][["audience", "porte", "pais", "genero"]].copy(); err["predicao"] = pred_test
    by_size = err.groupby("porte", observed=True).apply(lambda g: pd.Series(metrics(g.audience, g.predicao)), include_groups=False).reset_index()
    by_size.to_csv(OUT / "metricas_por_porte_modelo_unico.csv", index=False, encoding="utf-8-sig")
    err.assign(erro_abs=lambda z: abs(z.audience-z.predicao), erro_pct=lambda z: abs(z.audience-z.predicao)/z.audience*100).sort_values("erro_abs", ascending=False).to_csv(OUT / "erros_holdout.csv", index=False, encoding="utf-8-sig")

    # Estratégia segmentada-oráculo: usa o porte verdadeiro, logo serve só como diagnóstico e não como pipeline implantável.
    oracle_pred = np.full(len(test), np.nan)
    for size in df.porte.cat.categories:
        tr = train[df.porte.iloc[train].to_numpy() == size]
        te_pos = np.where(df.porte.iloc[test].to_numpy() == size)[0]
        if len(tr) >= 8 and len(te_pos):
            m = RandomForestRegressor(**grid.best_params_, random_state=SEED, n_jobs=-1).fit(X.iloc[tr], ylog.iloc[tr])
            oracle_pred[te_pos] = np.expm1(m.predict(X.iloc[test[te_pos]]))
    valid = ~np.isnan(oracle_pred)
    strategy = pd.DataFrame([
        {"estrategia": "A_modelo_unico", **metrics(df.audience.iloc[test], pred_test)},
        {"estrategia": "B_segmentado_oraculo", **metrics(df.audience.iloc[test[valid]], oracle_pred[valid])},
    ])
    strategy.to_csv(OUT / "comparacao_estrategias.csv", index=False, encoding="utf-8-sig")

    # Medição simples de inferência/memória do modelo final.
    tracemalloc.start(); start = time.perf_counter(); tuned.fit(X, ylog); fit_s = time.perf_counter()-start
    _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    batch = pd.concat([X] * 100, ignore_index=True)
    start = time.perf_counter(); tuned.predict(batch); infer_s = time.perf_counter()-start
    scalability = {"treino_s_205_linhas": fit_s, "pico_memoria_python_mb": peak/1024**2,
                   "inferencia_ms_por_linha": infer_s/len(batch)*1000, "linhas_batch": len(batch)}
    (OUT / "escalabilidade.json").write_text(json.dumps(scalability, indent=2), encoding="utf-8")
    print(comp.head(10).to_string(index=False))
    print("\nQualidade:", json.dumps(quality, ensure_ascii=False))
    print("\nEstratégias:\n", strategy.to_string(index=False))


if __name__ == "__main__":
    main()
