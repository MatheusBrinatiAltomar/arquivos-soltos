"""Gera uma única divisão treino/validação/teste para mmBERT e XGBoost hierárquico.

A partição usa os mesmos filtros e a mesma taxonomia de 49 áreas dos scripts
mmbert.py e treinar_hierarquico_balanceado.py. O resultado é um manifesto leve
com os índices das linhas originais do Parquet; não duplica o texto.

Execute na mesma pasta dos scripts de treinamento:
    python gerar_base_particionada.py
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORPUS = REPO / "dados" / "corpus_td_lemas.parquet"
DEFAULT_OUT = REPO / "dados" / "particoes_capes.csv"
DEFAULT_META = REPO / "dados" / "particoes_capes_meta.json"
SEED = 42

CONFIG_AREAS = {
    "CIÊNCIAS AGRÁRIAS": [
        "CIÊNCIA DE ALIMENTOS", "CIÊNCIAS AGRÁRIAS I", "MEDICINA VETERINÁRIA", "ZOOTECNIA / RECURSOS PESQUEIROS",
    ],
    "CIÊNCIAS BIOLÓGICAS": [
        "BIODIVERSIDADE", "CIÊNCIAS BIOLÓGICAS I", "CIÊNCIAS BIOLÓGICAS II", "CIÊNCIAS BIOLÓGICAS III",
    ],
    "CIÊNCIAS DA SAÚDE": [
        "EDUCAÇÃO FÍSICA", "ENFERMAGEM", "FARMÁCIA", "MEDICINA I", "MEDICINA II", "MEDICINA III", "NUTRIÇÃO", "ODONTOLOGIA", "SAÚDE COLETIVA",
    ],
    "CIÊNCIAS EXATAS E DA TERRA": [
        "ASTRONOMIA / FÍSICA", "CIÊNCIA DA COMPUTAÇÃO", "GEOCIÊNCIAS", "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA", "QUÍMICA",
    ],
    "CIÊNCIAS HUMANAS": [
        "ANTROPOLOGIA / ARQUEOLOGIA", "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS", "CIÊNCIAS DA RELIGIÃO E TEOLOGIA", "EDUCAÇÃO", "FILOSOFIA", "GEOGRAFIA", "HISTÓRIA", "PSICOLOGIA", "SOCIOLOGIA",
    ],
    "CIÊNCIAS SOCIAIS APLICADAS": [
        "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO", "ARQUITETURA, URBANISMO E DESIGN", "COMUNICAÇÃO E INFORMAÇÃO", "DIREITO", "ECONOMIA", "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA", "SERVIÇO SOCIAL",
    ],
    "ENGENHARIAS": ["ENGENHARIAS I", "ENGENHARIAS II", "ENGENHARIAS III", "ENGENHARIAS IV"],
    "LINGUÍSTICA, LETRAS E ARTES": ["ARTES", "LINGUÍSTICA E LITERATURA"],
    "MULTIDISCIPLINAR": ["BIOTECNOLOGIA", "CIÊNCIAS AMBIENTAIS", "ENSINO", "INTERDISCIPLINAR", "MATERIAIS"],
}
VALID_PAIRS = {(ga, classe) for ga, lista in CONFIG_AREAS.items() for classe in lista}


def normalizar_rotulo(valor: object) -> str:
    if pd.isna(valor):
        return ""
    texto = str(valor).replace("\u00a0", " ")
    texto = unicodedata.normalize("NFKC", texto)
    return " ".join(texto.strip().split()).upper()


def mapear_area_especialista(valor: object) -> str:
    """Mesma normalização de rótulos usada nos dois scripts de treinamento."""
    area = normalizar_rotulo(valor)
    mapa = {
        "ARTES / MÚSICA": "ARTES",
        "LINGUISTICA E LITERATURA": "LINGUÍSTICA E LITERATURA",
        "LETRAS / LINGUÍSTICA": "LINGUÍSTICA E LITERATURA",
        "LETRAS / LINGUISTICA": "LINGUÍSTICA E LITERATURA",
        "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS": "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS",
        "CIENCIA POLITICA E RELACOES INTERNACIONAIS": "CIÊNCIA POLÍTICA E RELAÇÕES INTERNACIONAIS",
        "CIÊNCIAS DA RELIGIÃO E TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "CIENCIAS DA RELIGIAO E TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "FILOSOFIA/TEOLOGIA:SUBCOMISSÃO TEOLOGIA": "CIÊNCIAS DA RELIGIÃO E TEOLOGIA",
        "EDUCAÇÃO": "EDUCAÇÃO",
        "EDUCACAO": "EDUCAÇÃO",
        "FILOSOFIA/TEOLOGIA:SUBCOMISSÃO FILOSOFIA": "FILOSOFIA",
        "HISTÓRIA": "HISTÓRIA",
        "HISTORIA": "HISTÓRIA",
        "ASTRONOMIA / FISICA": "ASTRONOMIA / FÍSICA",
        "ASTRONOMIA/FÍSICA": "ASTRONOMIA / FÍSICA",
        "COMPUTAÇÃO / CIÊNCIA DA COMPUTAÇÃO": "CIÊNCIA DA COMPUTAÇÃO",
        "COMPUTAÇÃO": "CIÊNCIA DA COMPUTAÇÃO",
        "CIENCIA DA COMPUTACAO": "CIÊNCIA DA COMPUTAÇÃO",
        "GEOCIENCIAS": "GEOCIÊNCIAS",
        "MATEMATICA / PROBABILIDADE E ESTATISTICA": "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA",
        "MATEMÁTICA / PROBABILIDADE ESTATÍSTICA": "MATEMÁTICA / PROBABILIDADE E ESTATÍSTICA",
        "QUIMICA": "QUÍMICA",
        "CIENCIAS BIOLOGICAS I": "CIÊNCIAS BIOLÓGICAS I",
        "CIENCIAS BIOLOGICAS II": "CIÊNCIAS BIOLÓGICAS II",
        "CIENCIAS BIOLOGICAS III": "CIÊNCIAS BIOLÓGICAS III",
        "EDUCACAO FISICA": "EDUCAÇÃO FÍSICA",
        "FARMACIA": "FARMÁCIA",
        "NUTRICAO": "NUTRIÇÃO",
        "ODONTOLOGIA": "ODONTOLOGIA",
        "SAUDE COLETIVA": "SAÚDE COLETIVA",
        "CIENCIAS AGRARIAS I": "CIÊNCIAS AGRÁRIAS I",
        "CIENCIA DE ALIMENTOS": "CIÊNCIA DE ALIMENTOS",
        "MEDICINA VETERINARIA": "MEDICINA VETERINÁRIA",
        "ZOOTECNIA / RECURSOS PESQUEIROS": "ZOOTECNIA / RECURSOS PESQUEIROS",
        "ZOOTECNIA/RECURSOS PESQUEIROS": "ZOOTECNIA / RECURSOS PESQUEIROS",
        "ADMINISTRACAO PUBLICA E DE EMPRESAS, CIENCIAS CONTABEIS E TURISMO": "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO",
        "ADMINISTRAÇÃO, CIÊNCIAS CONTÁBEIS E TURISMO": "ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO",
        "COMUNICACAO E INFORMACAO": "COMUNICAÇÃO E INFORMAÇÃO",
        "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA": "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA",
        "PLANEJAMENTO URBANO E REGIONAL/DEMOGRAFIA": "PLANEJAMENTO URBANO E REGIONAL / DEMOGRAFIA",
        "SERVICO SOCIAL": "SERVIÇO SOCIAL",
        "CIENCIAS AMBIENTAIS": "CIÊNCIAS AMBIENTAIS",
        "CIÊNCIA DOS MATERIAIS": "MATERIAIS",
    }
    return mapa.get(area, area)


def carregar_linhas_validas(batch_size: int) -> tuple[pd.DataFrame, int]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Instale pyarrow: pip install pyarrow") from exc

    if not CORPUS.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {CORPUS}")

    parquet = pq.ParquetFile(CORPUS)
    colunas = ["id_producao", "grande_area", "area_avaliacao", "lemmas_ext"]
    faltam = sorted(set(colunas) - set(parquet.schema.names))
    if faltam:
        raise ValueError(f"O corpus não contém as colunas necessárias: {faltam}")

    registros = []
    cursor = 0
    grandes_validas = set(CONFIG_AREAS)
    with tqdm(total=parquet.metadata.num_rows, desc="Lendo corpus", unit="docs", dynamic_ncols=True) as barra:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=colunas):
            bloco = batch.to_pandas()
            n = len(bloco)
            bloco["indice_parquet"] = np.arange(cursor, cursor + n, dtype=np.int64)
            cursor += n

            grande = bloco["grande_area"].map(normalizar_rotulo)
            classe = bloco["area_avaliacao"].map(mapear_area_especialista)
            texto_valido = bloco["lemmas_ext"].notna() & bloco["lemmas_ext"].astype(str).str.strip().ne("")
            par_valido = pd.Series(
                [(ga, cl) in VALID_PAIRS for ga, cl in zip(grande, classe)],
                index=bloco.index,
            )
            mask = grande.isin(grandes_validas) & texto_valido & par_valido
            if mask.any():
                sub = bloco.loc[mask, ["indice_parquet", "id_producao"]].copy()
                sub["grande_area"] = grande.loc[mask].to_numpy()
                sub["classe"] = classe.loc[mask].to_numpy()
                registros.append(sub)
            barra.update(n)

    if not registros:
        raise RuntimeError("Nenhuma linha válida; verifique rótulos, grande_area e lemmas_ext.")

    df = pd.concat(registros, ignore_index=True).sort_values("indice_parquet").reset_index(drop=True)
    df["indice_parquet"] = df["indice_parquet"].astype(np.int64)

    n_dup = int(df["id_producao"].dropna().duplicated().sum())
    if n_dup:
        print(
            f"AVISO: {n_dup:,} linhas válidas repetem id_producao. "
            "A divisão é feita por linha do Parquet; avalie possíveis duplicatas entre partições."
        )
    return df, int(parquet.metadata.num_rows)


def criar_particoes(df: pd.DataFrame, test_size: float, val_size: float, seed: int) -> pd.DataFrame:
    if test_size <= 0 or val_size <= 0 or test_size + val_size >= 1:
        raise ValueError("test-size e val-size devem ser > 0 e sua soma deve ser < 1.")

    # Estratifica por grande área + área de avaliação, para preservar cada
    # classe dentro do especialista correspondente.
    estrato = df["grande_area"].astype(str) + " || " + df["classe"].astype(str)
    contagens = estrato.value_counts()
    raras = contagens[contagens < 7]
    if not raras.empty:
        raise ValueError(
            "Cada par grande-área/área precisa ter ao menos 7 documentos para a divisão "
            "estratificada em duas etapas preservar exemplos em treino, validação e teste. "
            f"Pares insuficientes: {raras.to_dict()}"
        )

    idx = np.arange(len(df), dtype=np.int64)
    idx_treino, idx_temp = train_test_split(
        idx, test_size=test_size + val_size, random_state=seed, stratify=estrato.to_numpy()
    )
    frac_teste_temp = test_size / (test_size + val_size)
    idx_validacao, idx_teste = train_test_split(
        idx_temp, test_size=frac_teste_temp, random_state=seed, stratify=estrato.to_numpy()[idx_temp]
    )

    particao = np.empty(len(df), dtype=object)
    particao[idx_treino] = "treino"
    particao[idx_validacao] = "validacao"
    particao[idx_teste] = "teste"
    out = df.copy()
    out["particao"] = particao

    classe_ordem = [c for ga in CONFIG_AREAS for c in CONFIG_AREAS[ga]]
    classe_to_id = {c: i for i, c in enumerate(classe_ordem)}
    out["id_classe"] = out["classe"].map(classe_to_id).astype(np.int32)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(out))
    ordem = np.empty(len(out), dtype=np.int64)
    ordem[perm] = np.arange(len(out), dtype=np.int64)
    out["ordem_amostragem"] = ordem
    return out.sort_values("indice_parquet").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=50_000)
    args = parser.parse_args()

    df, n_original = carregar_linhas_validas(args.batch_size)
    particoes = criar_particoes(df, args.test_size, args.val_size, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.meta.parent.mkdir(parents=True, exist_ok=True)

    colunas = ["indice_parquet", "id_producao", "grande_area", "classe", "id_classe", "particao", "ordem_amostragem"]
    particoes[colunas].to_csv(args.out, index=False, encoding="utf-8")
    distribuicao = particoes.groupby(["particao", "grande_area", "classe"]).size().rename("quantidade").reset_index()
    totais = {str(k): int(v) for k, v in particoes["particao"].value_counts().items()}
    meta = {
        "seed": int(args.seed),
        "estrategia": "train_test_split estratificado em duas etapas por grande_area + classe",
        "proporcao_treino": round(1.0 - args.test_size - args.val_size, 6),
        "proporcao_validacao": float(args.val_size),
        "proporcao_teste": float(args.test_size),
        "n_linhas_parquet_original": n_original,
        "n_documentos_validos": int(len(particoes)),
        "n_classes": int(particoes["classe"].nunique()),
        "classes": sorted(particoes["classe"].unique().tolist()),
        "grandes_areas": sorted(particoes["grande_area"].unique().tolist()),
        "totais_por_particao": totais,
        "arquivo_corpus": str(CORPUS),
        "arquivo_particoes": str(args.out),
        "observacao": (
            "Taxonomia e filtros compartilhados pelos scripts mmBERT e XGBoost hierárquico balanceado. "
            "indice_parquet referencia a linha original do Parquet. O undersampling deve ocorrer somente no treino."
        ),
        "distribuicao_particao_classe": distribuicao.to_dict(orient="records"),
    }
    args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nParticionamento concluído.")
    print(f"CSV: {args.out}")
    print(f"Metadados: {args.meta}")
    print(f"Documentos válidos: {len(particoes):,}")
    for nome in ("treino", "validacao", "teste"):
        print(f"  {nome:10s}: {totais.get(nome, 0):,}")


if __name__ == "__main__":
    main()
