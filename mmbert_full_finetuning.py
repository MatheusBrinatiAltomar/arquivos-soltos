"""
Treina nove classificadores especializados baseados em mmBERT SEM LoRA/PEFT:
fine-tuning completo de TODOS os parâmetros do encoder e da cabeça de classificação.
As grandes áreas e as partições são compartilhadas com o experimento original.

Exemplos:
    python mmbert_full_finetuning.py --grande-area ciencias_biologicas
    python mmbert_full_finetuning.py --grande-area ciencias_humanas --max-len 128
    python mmbert_full_finetuning.py --grande-area "CIÊNCIAS DA SAÚDE" --optimizer adamw_torch

Recursos:
    - Fine-tuning completo, gradient checkpointing e precisão mista CUDA.
    - Otimizador Adafactor por padrão para reduzir VRAM; AdamW opcional.
    - Early stopping e melhor checkpoint selecionados por eval_loss.
    - Extração de perda por documento e métricas de teste.
    - Atribuição de tokens com Captum opcional (--captum), intensiva em memória.
    - Saída isolada de experimentos com LoRA.

Instalação:
    pip install -U "transformers>=4.57.0" torch datasets accelerate scikit-learn pandas pyarrow tqdm
    pip install -U captum  # opcional, somente para --captum

ATENÇÃO: full fine-tuning de ~307 milhões de parâmetros consome muito mais VRAM
que LoRA. Uma GPU de 6 GB pode apresentar CUDA out of memory mesmo com
batch-size 1, gradient checkpointing, Adafactor e max-len reduzido.
"""

from __future__ import annotations

import argparse
import gc
from contextlib import nullcontext
import json
import os
import sys
import time
import unicodedata

# Deve ser definido antes de importar torch. O recurso expandable_segments não é
# suportado por builds do PyTorch no Windows; não o habilitamos nessa plataforma.
if os.name != "nt":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
else:
    # Se a variável foi configurada fora do script, remove apenas essa opção
    # incompatível com Windows e preserva as demais opções do alocador.
    for _var_cuda in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        _cfg_cuda = os.environ.get(_var_cuda, "")
        if _cfg_cuda and "expandable_segments" in _cfg_cuda:
            _parts_cuda = [
                part for part in _cfg_cuda.split(",")
                if not part.strip().lower().startswith("expandable_segments")
            ]
            if _parts_cuda:
                os.environ[_var_cuda] = ",".join(_parts_cuda)
            else:
                os.environ.pop(_var_cuda, None)
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    set_seed
)

# Tentativa de importação opcional do Captum para interpretabilidade de tokens
try:
    from captum.attr import LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False


# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
CORPUS = REPO / "dados" / "corpus_td_lemas.parquet"
T2A_DATA = REPO / "texto2area" / "data"
T2A_VECTORIZER = T2A_DATA / "vetorizador.joblib"
T2A_MODEL = T2A_DATA / "modelo.joblib"
OUT_DIR = HERE / "modelo_treinado" / "mmbert_full_finetuning"
PARTICOES_FILE = REPO / "dados" / "particoes_capes.csv"


# DATA_DIR = Path(
#     "/kaggle/input/datasets/mthaltomar97/"
#     "dataset-capes/meu-dataset-capes/dados"
# )

# CORPUS = DATA_DIR / "corpus_td_lemas.parquet"
# PARTICOES_FILE = DATA_DIR / "particoes_capes.csv"



SEED = 42
set_seed(SEED)

# ---------------------------------------------------------------------------
# Taxonomia (49 Áreas de Avaliação da CAPES)
# ---------------------------------------------------------------------------

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
    "ENGENHARIAS": [
        "ENGENHARIAS I", "ENGENHARIAS II", "ENGENHARIAS III", "ENGENHARIAS IV",
    ],
    "LINGUÍSTICA, LETRAS E ARTES": [
        "ARTES", "LINGUÍSTICA E LITERATURA",
    ],
    "MULTIDISCIPLINAR": [
        "BIOTECNOLOGIA", "CIÊNCIAS AMBIENTAIS", "ENSINO", "INTERDISCIPLINAR", "MATERIAIS",
    ],
}

SLUGS = {
    "CIÊNCIAS AGRÁRIAS": "ciencias_agrarias",
    "CIÊNCIAS BIOLÓGICAS": "ciencias_biologicas",
    "CIÊNCIAS DA SAÚDE": "ciencias_da_saude",
    "CIÊNCIAS EXATAS E DA TERRA": "ciencias_exatas_e_da_terra",
    "CIÊNCIAS HUMANAS": "ciencias_humanas",
    "CIÊNCIAS SOCIAIS APLICADAS": "ciencias_sociais_aplicadas",
    "ENGENHARIAS": "engenharias",
    "LINGUÍSTICA, LETRAS E ARTES": "linguistica_letras_artes",
    "MULTIDISCIPLINAR": "multidisciplinar",
}

VALID_PAIRS = {(ga, classe) for ga, classes in CONFIG_AREAS.items() for classe in classes}

DEFAULT_MMBERT_MODEL = "jhu-clsp/mmBERT-base"
DEFAULT_MAX_LEN = 256
DEFAULT_BATCH = 1
DEFAULT_GRADIENT_ACCUMULATION = 16
DEFAULT_EPOCHS = 4
DEFAULT_LR = 2e-5
DEFAULT_OPTIMIZER = "adafactor"  # Menos memória que AdamW, adequado ao full fine-tuning.


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def normalizar_rotulo(valor: object) -> str:
    if pd.isna(valor):
        return ""
    texto = str(valor).replace("\u00a0", " ")
    texto = unicodedata.normalize("NFKC", texto)
    return " ".join(texto.strip().split()).upper()

def mapear_area_especialista(valor: object) -> str:
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

def titulo(texto: str) -> None:
    print("\n" + "=" * 78)
    print(texto)
    print("=" * 78, flush=True)


# ---------------------------------------------------------------------------
# Carregamento de Dados e Balanceamento
# ---------------------------------------------------------------------------

def carregar_corpus(
    amostra: int | None,
    grande_area: str | None = None,
) -> pd.DataFrame:
    """Carrega apenas as grandes áreas solicitadas e herda a partição compartilhada.

    `indice_parquet` é a posição da linha no corpus original. A associação por
    esse índice evita que a ordem após filtros altere a partição de um documento.
    """
    import pyarrow.parquet as pq

    if not CORPUS.exists():
        raise FileNotFoundError(f"Corpus não encontrado: {CORPUS}")
    if not PARTICOES_FILE.exists():
        raise FileNotFoundError(
            f"Arquivo de partições não encontrado: {PARTICOES_FILE}\n"
            "Execute primeiro: python gerar_base_particionada.py"
        )

    particoes = pd.read_csv(
        PARTICOES_FILE,
        usecols=["indice_parquet", "classe", "particao", "ordem_amostragem"],
    )
    particoes["indice_parquet"] = particoes["indice_parquet"].astype(np.int64)
    if amostra is not None:
        if amostra < 1:
            raise ValueError("--amostra precisa ser >= 1")
        particoes = particoes.loc[particoes["ordem_amostragem"] < amostra].copy()

    pf = pq.ParquetFile(CORPUS)
    colunas = ["id_producao", "grande_area", "area_avaliacao", "lemmas_ext"]
    partes = []
    if grande_area is not None:
        if grande_area not in CONFIG_AREAS:
            raise ValueError(f"Grande área não reconhecida: {grande_area}")
        grandes_validas = {grande_area}
    else:
        grandes_validas = set(CONFIG_AREAS.keys())
    cursor_original = 0

    with tqdm(total=pf.metadata.num_rows, desc="Lendo corpus", unit=" docs") as bar:
        for tabela in pf.iter_batches(batch_size=50000, columns=colunas):
            bloco = tabela.to_pandas()
            n_bloco = len(bloco)
            bloco["indice_parquet"] = np.arange(
                cursor_original, cursor_original + n_bloco, dtype=np.int64
            )
            cursor_original += n_bloco

            grande = bloco["grande_area"].map(normalizar_rotulo)
            bloco = bloco.loc[grande.isin(grandes_validas)].copy()
            if bloco.empty:
                bar.update(n_bloco)
                continue

            bloco["grande_area"] = grande.loc[bloco.index]
            bloco = bloco.dropna(subset=["lemmas_ext"])
            bloco = bloco[bloco["lemmas_ext"].astype(str).str.strip() != ""]
            if not bloco.empty:
                partes.append(bloco)
            bar.update(n_bloco)

    if not partes:
        raise RuntimeError("Nenhum documento encontrado após os filtros do mmBERT.")

    df = pd.concat(partes, ignore_index=True)
    df = df.merge(
        particoes[["indice_parquet", "classe", "particao", "ordem_amostragem"]],
        on="indice_parquet",
        how="inner",
        validate="one_to_one",
    )
    if df.empty:
        raise RuntimeError(
            "Nenhum documento do mmBERT corresponde ao arquivo de partições. "
            "Recrie particoes_capes.csv com gerar_base_particionada.py."
        )

    df = df.sort_values("indice_parquet").reset_index(drop=True)
    # O gerador já filtrou os pares válidos e padronizou os rótulos. Assim,
    # o campo do manifesto é a fonte canônica usada pelos dois modelos.
    df["area_classe"] = df["classe"].astype(str).str.strip().str.upper()
    mask_pares = pd.Series(
        [(ga, ac) in VALID_PAIRS for ga, ac in zip(df["grande_area"], df["area_classe"])],
        index=df.index,
    )
    if not mask_pares.all():
        raise ValueError("O manifesto compartilhado contém pares grande-área/área que não constam em CONFIG_AREAS do mmBERT.")
    df = df.drop(columns=["classe"])
    print(f"Documentos comuns aos modelos, com partição compartilhada: {len(df):,}")
    print("Partições no corpus mmBERT:")
    for nome in ("treino", "validacao", "teste"):
        print(f"  {nome:10s}: {(df['particao'] == nome).sum():,}")
    if grande_area is not None:
        print(f"Filtro de grande área aplicado: {grande_area}")
    if amostra is not None:
        print(f"Subamostra compartilhada solicitada: ordem_amostragem < {amostra:,}")
    return df



def balancear_dataset_por_undersampling(y: np.ndarray, n_classes: int) -> tuple:
    """Balanceia apenas classes que realmente aparecem no treino selecionado.

    Isso permite executar subamostras de teste sem falhar só porque uma classe
    rara ficou fora da amostra. Em treino completo, as classes presentes são
    balanceadas pelo menor suporte entre elas.
    """
    rng = np.random.default_rng(SEED)
    contagens = np.array([np.sum(y == classe) for classe in range(n_classes)])
    classes_presentes = np.flatnonzero(contagens > 0)
    if len(classes_presentes) < 2:
        raise ValueError("O treino precisa conter ao menos duas classes para aplicar undersampling.")
    alvo = int(contagens[classes_presentes].min())

    partes = []
    registros = []

    for classe in classes_presentes:
        idx_classe = np.flatnonzero(y == classe)
        escolhidos = rng.choice(idx_classe, size=alvo, replace=False)
        partes.append(escolhidos)
        registros.append({
            "classe_id": int(classe), "quantidade_antes": len(idx_classe),
            "quantidade_depois": alvo, "alvo_balanceamento": alvo,
        })

    idx_balanceado = np.concatenate(partes)
    rng.shuffle(idx_balanceado)
    return idx_balanceado, pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# Extração Detalhada de Resíduos (Loss por documento)
# ---------------------------------------------------------------------------

def extrair_residuos_mmbert(
    model: nn.Module,
    ds_test: Dataset,
    textos_test: np.ndarray,
    classes: list[str],
    tokenizer,
    batch_size: int = DEFAULT_BATCH,
    precisao: str = "fp32",
) -> pd.DataFrame:
    """
    Calcula a perda (Cross-Entropy) por documento individual sem redução,
    gerando a tabela de resíduos exata para comparar com o XGBoost.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="none")
    
    # O collator aplica padding dinâmico aos documentos de cada lote.
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )
    dataloader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    all_losses = []
    all_preds = []
    all_probs = []
    all_prob_true = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="A extrair resíduos do mmBERT", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Mesma precisão mista da avaliação do Trainer; saída FP32 para a loss.
            with autocast_inferencia(precisao):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float()

            # Perda isolada por documento em FP32 para maior estabilidade numérica.
            losses = criterion(logits, labels)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            prob_true = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

            all_losses.extend(losses.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_prob_true.extend(prob_true.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    probs_matrix = np.vstack(all_probs)
    
    df_residuos = pd.DataFrame({
        "texto": textos_test,
        "y_true": [classes[i] for i in all_labels],
        "y_pred": [classes[i] for i in all_preds],
        "loss_mmbert": all_losses,
        "confianca": probs_matrix.max(axis=1),
        "prob_classe_correta": all_prob_true,
    })

    return df_residuos


def extrair_atribuicao_tokens_captum(
    model: nn.Module,
    tokenizer,
    texto: str,
    label_id: int,
    caminho_saida: Path,
    max_len: int = DEFAULT_MAX_LEN,
    precisao: str = "fp32",
) -> None:
    """
    Extrai a importância de cada token usando Layer Integrated Gradients do Captum.
    """
    if not HAS_CAPTUM:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def predict_func(inputs, attention_mask):
        with autocast_inferencia(precisao):
            return model(
                input_ids=inputs,
                attention_mask=attention_mask,
            ).logits.float()

    # Lê os embeddings diretamente do modelo Hugging Face, sem wrappers/adaptadores.
    embedding_layer = model.get_input_embeddings()
    if embedding_layer is None:
        print("[AVISO] Não foi possível localizar embeddings para o Captum; atribuição ignorada.")
        return
    lig = LayerIntegratedGradients(predict_func, embedding_layer)

    inputs = tokenizer(
        texto, return_tensors="pt", truncation=True, max_length=max_len
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # N_steps menor + internal_batch_size=1 reduzem picos de VRAM.
    attributions, delta = lig.attribute(
        inputs=input_ids,
        additional_forward_args=(attention_mask,),
        target=label_id,
        n_steps=16,
        internal_batch_size=1,
        return_convergence_delta=True,
    )

    # Soma a atribuição ao longo da dimensão de embedding
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    attr_scores = attributions.sum(dim=-1).squeeze(0).cpu().numpy()

    df_attr = pd.DataFrame({"token": tokens, "importance": attr_scores})
    df_attr.sort_values(by="importance", ascending=False).to_csv(caminho_saida, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Fine-tuning completo do mmBERT (especialista)
# ---------------------------------------------------------------------------

def resolver_precisao(selecao: str) -> str:
    """Usa AMP sem reduzir os parâmetros treináveis para FP16.

    A cópia principal dos pesos e os gradientes ficam em FP32; o Trainer controla
    a precisão mista durante o forward/backward, evitando o erro de GradScaler
    ao tentar fazer unscale de gradientes FP16.
    """
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    if selecao == "auto":
        return "bf16" if bf16 else ("fp16" if cuda else "fp32")
    if selecao == "bf16" and not bf16:
        raise ValueError("--precision bf16 requer GPU CUDA compatível com BF16.")
    if selecao == "fp16" and not cuda:
        raise ValueError("--precision fp16 requer GPU CUDA.")
    return selecao


def autocast_inferencia(precisao: str):
    if torch.cuda.is_available() and precisao in ("fp16", "bf16"):
        dtype = torch.float16 if precisao == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    bal_acc = balanced_accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }

def treinar_especialista_mmbert(
    nome: str,
    textos: np.ndarray,
    labels: np.ndarray,
    particoes: np.ndarray,
    indices_parquet: np.ndarray,
    ids_producao: np.ndarray,
    classes: list[str],
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    titulo(f"mmBERT — {nome.upper()}")
    
    classes_to_int = {c: i for i, c in enumerate(classes)}
    y_original = np.asarray([classes_to_int[str(v)] for v in labels], dtype=np.int64)

    print(f"Documentos originais totais: {len(y_original):,}")
    print(f"Classes:                     {len(classes)}")

    # 1) Reutiliza a partição global criada por gerar_base_particionada.py.
    if len(particoes) != len(y_original):
        raise ValueError("O vetor de partições não tem o mesmo tamanho dos rótulos do especialista.")
    if len(indices_parquet) != len(y_original) or len(ids_producao) != len(y_original):
        raise ValueError("Os identificadores do Parquet não têm o mesmo tamanho dos dados do especialista.")
    idx_train = np.flatnonzero(particoes == "treino")
    idx_val = np.flatnonzero(particoes == "validacao")
    idx_test = np.flatnonzero(particoes == "teste")
    if min(len(idx_train), len(idx_val), len(idx_test)) == 0:
        raise ValueError(
            f"A partição compartilhada não contém treino/validação/teste suficientes para {nome}."
        )
    print(
        f"Partição compartilhada -> Treino: {len(idx_train):,} | "
        f"Validação: {len(idx_val):,} | Teste: {len(idx_test):,}"
    )

    textos_train_orig = textos[idx_train]
    y_train_orig = y_original[idx_train]
    textos_val = textos[idx_val]
    y_val = y_original[idx_val]
    textos_test = textos[idx_test]
    y_test = y_original[idx_test]

    # 2) Undersampling no treino
    if args.balancear_treino:
        idx_balanceado, balanceamento = balancear_dataset_por_undersampling(y_train_orig, len(classes))
        textos_train = textos_train_orig[idx_balanceado]
        y_train = y_train_orig[idx_balanceado]
        print(f"Total no treino após balanceamento (Undersampling): {len(y_train):,} documentos")
    else:
        textos_train = textos_train_orig
        y_train = y_train_orig
        print("Balanceamento por undersampling: DESATIVADO")
        balanceamento = pd.DataFrame()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Fine-tuning COMPLETO: todos os pesos-base + classificador são treináveis.
    # Os pesos são carregados em FP32 (parâmetros de referência); a precisão
    # mista AMP é aplicada pelo Trainer, sem converter os parâmetros para FP16.
    # Isso evita "Attempting to unscale FP16 gradients" do GradScaler.
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(classes),
        id2label={i: c for i, c in enumerate(classes)},
        label2id=classes_to_int,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    n_total = sum(p.numel() for p in model.parameters())
    n_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros totais:     {n_total:,}")
    print(f"Parâmetros treináveis: {n_treinaveis:,} ({100*n_treinaveis/n_total:.2f}%)")
    if n_total != n_treinaveis:
        raise RuntimeError("Fine-tuning completo esperado, mas há parâmetros congelados.")

    precisao = resolver_precisao(args.precision)
    print(f"Precisão de treino: {precisao.upper()} (parâmetros em FP32)")
    print(f"Otimizador:        {args.optimizer}")
    if args.gradient_checkpointing and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    def criar_dataset_formatado(t_array, y_array):
        data = {"text": t_array.tolist(), "labels": y_array.tolist()}
        dataset = Dataset.from_dict(data)
        # O padding é feito dinamicamente por lote, reduzindo o consumo de VRAM
        # em comparação com preencher todos os documentos até max_len.
        return dataset.map(
            lambda x: tokenizer(
                x["text"], truncation=True, padding=False, max_length=args.max_len
            ),
            batched=True,
            remove_columns=["text"],
            desc="Tokenizando documentos",
        )

    ds_train = criar_dataset_formatado(textos_train, y_train)
    ds_val = criar_dataset_formatado(textos_val, y_val)
    ds_test = criar_dataset_formatado(textos_test, y_test)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    model_dir = out_dir / nome
    model_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(model_dir / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_accumulation_steps=1,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        optim=args.optimizer,
        load_best_model_at_end=True,
        # Early stopping e seleção do melhor checkpoint pela menor perda de validação.
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=50,
        # AMP somente nas operações. O otimizador mantém pesos treináveis FP32.
        fp16=(precisao == "fp16"),
        bf16=(precisao == "bf16"),
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nIniciando fine-tuning do mmBERT...")
    t0 = time.time()
    trainer.train()
    tempo_treino = time.time() - t0

    print("\nAvaliando e gerando análise de resíduos detalhada...")
    df_residuos = extrair_residuos_mmbert(
        model=model,
        ds_test=ds_test,
        textos_test=textos_test,
        classes=classes,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        precisao=precisao,
    )
    # Identificadores que permitem alinhar exatamente esta previsão ao XGBoost.
    df_residuos.insert(0, "indice_parquet", indices_parquet[idx_test])
    df_residuos.insert(1, "id_producao", ids_producao[idx_test])

    # Guarda o ficheiro de resíduos individuais para comparar diretamente com o XGBoost
    df_residuos.to_csv(model_dir / "predicoes_residuos_mmbert.csv", index=False, encoding="utf-8-sig")

    # [Opcional] Atribuição de importância dos tokens para o caso de maior erro (maior resíduo)
    if args.captum and HAS_CAPTUM and not df_residuos.empty:
        pior_caso = df_residuos.sort_values(by="loss_mmbert", ascending=False).iloc[0]
        label_id = classes_to_int[pior_caso["y_true"]]
        try:
            extrair_atribuicao_tokens_captum(
                model=model,
                tokenizer=tokenizer,
                texto=str(pior_caso["texto"]),
                label_id=label_id,
                caminho_saida=model_dir / "importancia_tokens_maior_residuo.csv",
                max_len=args.max_len,
                precisao=precisao,
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print("[AVISO] Sem VRAM para Integrated Gradients; pulando a atribuição de tokens.")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.captum and not HAS_CAPTUM:
        print("[AVISO] --captum solicitado, mas Captum não está instalado.")

    # Avaliação agregada
    test_results = trainer.predict(ds_test)
    y_pred = np.argmax(test_results.predictions, axis=-1)
    y_true = ds_test["labels"]
    metrics = test_results.metrics
    
    # Salvamento de artefatos
    model_dir.mkdir(parents=True, exist_ok=True)
    pasta_modelo_final = model_dir / "modelo_final"
    pasta_modelo_final.mkdir(parents=True, exist_ok=True)

    # Salva o modelo COMPLETO (encoder + classificador), sem merge ou adaptadores.
    # O melhor checkpoint já foi restaurado pelo Trainer (menor eval_loss).
    modelo_final = trainer.model
    modelo_final.to("cpu")
    modelo_final.save_pretrained(str(pasta_modelo_final), safe_serialization=True)
    tokenizer.save_pretrained(str(pasta_modelo_final))
    del modelo_final
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not balanceamento.empty:
        balanceamento["classe"] = [classes[int(i)] for i in balanceamento["classe_id"]]
        balanceamento.to_csv(model_dir / "balanceamento_treino.csv", index=False, encoding="utf-8-sig")

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(model_dir / "matriz_confusao.csv", encoding="utf-8-sig")

    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(model_dir / "classification_report.csv", encoding="utf-8-sig")

    meta = {
        "nome": nome,
        "classes": classes,
        "n_documentos": int(len(y_original)),
        "n_treino": int(len(y_train)),
        "n_validacao": int(len(y_val)),
        "n_teste": int(len(y_test)),
        "test_accuracy": float(metrics["test_accuracy"]),
        "test_balanced_accuracy": float(metrics["test_balanced_accuracy"]),
        "test_f1_macro": float(metrics["test_f1_macro"]),
        "test_f1_weighted": float(metrics["test_f1_weighted"]),
        "segundos_treino": round(tempo_treino, 3),
        "modelo_base": args.model_name,
        "metodo_ajuste": "fine_tuning_completo",
        "parametros_totais": int(n_total),
        "parametros_treinaveis": int(n_treinaveis),
        "otimizador": args.optimizer,
        "precisao": precisao,
        "melhor_eval_loss": (
            float(trainer.state.best_metric) if trainer.state.best_metric is not None else None
        ),
        "melhor_checkpoint": trainer.state.best_model_checkpoint,
        "max_len": int(args.max_len),
        "batch_size": int(args.batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "params_treino": training_args.to_dict()
    }
    
    (model_dir / "meta_treino.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Finalizado {nome}. Acurácia de teste: {metrics['test_accuracy']:.4f}")

    return {"model_path": str(model_dir / "modelo_final"), "classes": classes, "metadata": meta}


# ---------------------------------------------------------------------------
# Inferência Completa
# ---------------------------------------------------------------------------

def testar_texto_mmbert(
    texto: str, diretorios_modelos: dict, max_len: int = DEFAULT_MAX_LEN
) -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from texto2area import classificar

    area, margens, termos = classificar(texto, topo=5)
    
    titulo("PREDIÇÃO EM TEXTO CRU (INFERÊNCIA)")
    print(f"Grande área (Routing texto2area): {area}")

    chave = SLUGS.get(area)
    if not chave or chave not in diretorios_modelos:
        print(f"\nNenhum modelo especialista mmBERT encontrado para '{area}'.")
        return

    caminho_modelo = diretorios_modelos[chave]
    print(f"Carregando especialista mmBERT de: {caminho_modelo}")
    
    tokenizer = AutoTokenizer.from_pretrained(caminho_modelo)
    model = AutoModelForSequenceClassification.from_pretrained(caminho_modelo)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=max_len)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
        pred_id = np.argmax(probs)
    
    classe_prevista = model.config.id2label[pred_id]
    
    print(f"\nEspecialista acionado: {chave}")
    print(f"Área de avaliação prevista pelo mmBERT: {classe_prevista}")
    print("\nProbabilidades Top 3:")
    
    classes_rankeadas = [(model.config.id2label[i], probs[i]) for i in range(len(probs))]
    classes_rankeadas.sort(key=lambda x: x[1], reverse=True)
    
    for c, p in classes_rankeadas[:3]:
        print(f"  {c}: {p:.4%}")


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def converter_grande_area(valor: str) -> str:
    """Aceita o nome oficial de uma grande área ou seu slug sem acentos."""
    valor_original = valor.strip()
    valor_normalizado = normalizar_rotulo(valor_original)
    for grande_area, slug in SLUGS.items():
        if valor_normalizado == normalizar_rotulo(grande_area) or valor_original.casefold() == slug.casefold():
            return grande_area
    opcoes = ", ".join(SLUGS.values())
    raise argparse.ArgumentTypeError(
        f"Grande área inválida: {valor!r}. Use o nome oficial ou um destes identificadores: {opcoes}"
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--amostra", type=int, default=None)
    ap.add_argument(
        "--grande-area",
        type=converter_grande_area,
        default=None,
        metavar="NOME_OU_SLUG",
        help=(
            "Treina apenas uma grande área. Aceita o nome oficial (entre aspas, se tiver espaços) "
            "ou um identificador como ciencias_biologicas. Sem esta opção, treina todas as áreas."
        ),
    )
    ap.add_argument(
        "--model-name", "--mmbert-model", "--bert-model",
        dest="model_name", type=str, default=DEFAULT_MMBERT_MODEL,
        help="Identificador Hugging Face do modelo (padrão: jhu-clsp/mmBERT-base).",
    )
    ap.add_argument(
        "--max-len", type=int, default=DEFAULT_MAX_LEN,
        help="Limite de tokens por documento. O padrão 256 reduz bastante o uso de VRAM.",
    )
    ap.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH,
        help="Lote por dispositivo (padrão 1 para GPU com 6 GB).",
    )
    ap.add_argument(
        "--gradient-accumulation-steps", type=int, default=DEFAULT_GRADIENT_ACCUMULATION,
        help="Acumulação de gradientes; lote efetivo = batch-size x acumulação.",
    )
    ap.add_argument(
        "--optimizer", choices=["adafactor", "adamw_torch"], default=DEFAULT_OPTIMIZER,
        help="Adafactor economiza memória (padrão); AdamW permite comparação de otimizadores.",
    )
    ap.add_argument(
        "--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
        help="Precisão mista do treino (auto: BF16 se disponível; senão FP16/FP32).",
    )
    ap.add_argument(
        "--output-dir", type=Path, default=OUT_DIR,
        help="Diretório de saída separado dos experimentos com LoRA.",
    )
    ap.add_argument(
        "--captum", action="store_true",
        help="Executa Integrated Gradients para o maior resíduo; desativado por padrão para poupar VRAM.",
    )
    ap.add_argument(
        "--sem-gradient-checkpointing", dest="gradient_checkpointing",
        action="store_false", help="Desativa gradient checkpointing (consome mais VRAM).",
    )
    ap.set_defaults(gradient_checkpointing=True)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    ap.add_argument("--sem-balanceamento", dest="balancear_treino", action="store_false")
    ap.set_defaults(balancear_treino=True)
    ap.add_argument("--texto", type=str, default=None)
    ap.add_argument("--sem-treinamento", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size precisa ser >= 1")
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps precisa ser >= 1")
    if args.max_len < 8:
        raise SystemExit("--max-len precisa ser >= 8")
    if args.epochs <= 0:
        raise SystemExit("--epochs precisa ser > 0")
    if args.learning_rate <= 0:
        raise SystemExit("--learning-rate precisa ser > 0")
    inicio_total = time.time()
    out_dir = args.output_dir

    titulo("HIERARQUIA COM mmBERT — FINE-TUNING COMPLETO (SEM LoRA)")
    print(f"Modelo Base: {args.model_name}")
    print(f"Device:      {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"Otimizador:  {args.optimizer}")
    print(f"Saída:       {out_dir}")
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"VRAM:        {vram:.1f} GiB")
        if vram < 12:
            print("[AVISO] Fine-tuning completo pode estourar a VRAM. "
                  "Tente --max-len 128; não há garantia de caber em GPUs de 6 GB.")
    if args.grande_area is None:
        print("Grandes áreas: todas (comportamento padrão)")
    else:
        print(f"Grande área selecionada: {args.grande_area} [{SLUGS[args.grande_area]}]")

    if args.sem_treinamento:
        if not args.texto:
            raise SystemExit("--sem-treinamento exige argumento --texto")
        
        dir_modelos = {}
        for ga, slug in SLUGS.items():
            path_modelo = out_dir / slug / "modelo_final"
            if path_modelo.exists():
                dir_modelos[slug] = str(path_modelo)
        
        if not dir_modelos:
            raise FileNotFoundError(f"Nenhum modelo mmBERT encontrado em {out_dir}")
            
        testar_texto_mmbert(args.texto, dir_modelos, max_len=args.max_len)
        return

    titulo("1/3 — CARREGAMENTO DO CORPUS")
    df = carregar_corpus(args.amostra, args.grande_area)
    print(f"Documentos selecionados: {len(df):,}")

    titulo("2/3 — TREINAMENTO DOS ESPECIALISTAS (full fine-tuning)")
    out_dir.mkdir(parents=True, exist_ok=True)

    for ga, classes in CONFIG_AREAS.items():
        if args.grande_area is not None and ga != args.grande_area:
            continue

        chave = SLUGS[ga]
        mask_ga = df["grande_area"].eq(ga).to_numpy()

        if not mask_ga.any():
            print(f"\nAviso: Nenhuma amostra encontrada para {ga}. Pulando...")
            continue

        textos_ga = df.loc[mask_ga, "lemmas_ext"].to_numpy()
        labels_ga = df.loc[mask_ga, "area_classe"].to_numpy()
        particoes_ga = df.loc[mask_ga, "particao"].to_numpy()
        indices_parquet_ga = df.loc[mask_ga, "indice_parquet"].to_numpy(dtype=np.int64)
        ids_producao_ga = df.loc[mask_ga, "id_producao"].to_numpy()

        try:
            res = treinar_especialista_mmbert(
                chave, textos_ga, labels_ga, particoes_ga,
                indices_parquet_ga, ids_producao_ga, classes, out_dir, args
            )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"\n[ERRO] Falha ao treinar especialista {ga}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    titulo("3/3 — RESUMO FINAL")
    # Recria o resumo a partir dos metadados persistidos de todos os especialistas
    # concluídos. Assim, executar uma única grande área não apaga os resultados
    # das áreas treinadas em execuções anteriores.
    registros_resumo = []
    for grande_area, slug in SLUGS.items():
        caminho_meta = out_dir / slug / "meta_treino.json"
        if not caminho_meta.exists():
            continue
        try:
            meta_salva = json.loads(caminho_meta.read_text(encoding="utf-8"))
            registros_resumo.append({
                "especialista": grande_area,
                "n_treino": meta_salva.get("n_treino"),
                "f1_macro": meta_salva.get("test_f1_macro"),
                "accuracy": meta_salva.get("test_accuracy"),
            })
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[AVISO] Não foi possível ler {caminho_meta}: {exc}")

    if registros_resumo:
        df_resumo = pd.DataFrame(registros_resumo)
        ordem_areas = {ga: i for i, ga in enumerate(CONFIG_AREAS)}
        df_resumo["_ordem"] = df_resumo["especialista"].map(ordem_areas)
        df_resumo = df_resumo.sort_values("_ordem").drop(columns="_ordem")
        print(df_resumo.to_string(index=False))
        caminho_resumo = out_dir / "resumo_experimento_mmbert_full.csv"
        df_resumo.to_csv(caminho_resumo, index=False, encoding="utf-8-sig")
        print(f"\nResumo acumulado salvo em: {caminho_resumo}")
    else:
        print("Nenhum especialista foi concluído nesta execução.")

    print(f"\nTempo total: {time.time() - inicio_total:,.1f} s")

    if args.texto:
        dir_modelos = {
            slug: str(out_dir / slug / "modelo_final") for slug in SLUGS.values()
            if (out_dir / slug / "modelo_final").exists()
        }
        testar_texto_mmbert(args.texto, dir_modelos, max_len=args.max_len)

if __name__ == "__main__":
    main()