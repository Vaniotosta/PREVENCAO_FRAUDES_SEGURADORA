
from pathlib import Path

import numpy as np
import pandas as pd


# Caminho principal do projeto
ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SINISTROS = 20_000


def gerar_dataset(
    n_sinistros=N_SINISTROS,
    random_state=RANDOM_STATE,
):
    """Gera dados artificiais de sinistros de seguros."""

    rng = np.random.default_rng(random_state)

    idade = rng.integers(18, 85, n_sinistros)
    tempo_cliente = rng.integers(0, 25, n_sinistros)

    tipo_seguro = rng.choice(
        ["Automóvel", "Residencial", "Vida", "Empresarial"],
        n_sinistros,
        p=[0.65, 0.15, 0.10, 0.10],
    )

    regiao = rng.choice(
        ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"],
        n_sinistros,
    )

    valor_sinistro = rng.lognormal(
        mean=8.5, sigma=1.0, size=n_sinistros
    )

    valor_apolice = rng.lognormal(
        mean=9.0, sigma=0.7, size=n_sinistros
    )

    sinistros_anteriores = rng.poisson(0.7, n_sinistros)
    dias_desde_contratacao = rng.integers(1, 3650, n_sinistros)
    hora_ocorrencia = rng.integers(0, 24, n_sinistros)
    dias_para_comunicar = rng.integers(0, 31, n_sinistros)

    tipo_ocorrencia = rng.choice(
        ["Colisão", "Roubo", "Incêndio", "Dano climático", "Outros"],
        n_sinistros,
    )

    canal_aviso = rng.choice(
        ["Aplicativo", "Telefone", "Corretor", "Site"],
        n_sinistros,
    )

    # Rótulo sintético para fins didáticos
    logit = (
        -3.5
        + 0.65 * (sinistros_anteriores >= 3)
        + 0.55 * (dias_desde_contratacao < 60)
        + 0.45 * (valor_sinistro > np.quantile(valor_sinistro, 0.90))
        + 0.35 * (dias_para_comunicar > 10)
        + 0.25 * (hora_ocorrencia <= 4)
    )

    probabilidade_fraude = 1 / (1 + np.exp(-logit))
    fraude_confirmada = rng.binomial(1, probabilidade_fraude)

    df = pd.DataFrame(
        {
            "id_sinistro": [
                f"SIN{i:07d}" for i in range(1, n_sinistros + 1)
            ],
            "idade_segurado": idade,
            "tempo_cliente_anos": tempo_cliente,
            "tipo_seguro": tipo_seguro,
            "regiao": regiao,
            "valor_sinistro": np.round(valor_sinistro, 2),
            "valor_apolice": np.round(valor_apolice, 2),
            "sinistros_anteriores": sinistros_anteriores,
            "dias_desde_contratacao": dias_desde_contratacao,
            "tipo_ocorrencia": tipo_ocorrencia,
            "canal_aviso": canal_aviso,
            "hora_ocorrencia": hora_ocorrencia,
            "dias_para_comunicar": dias_para_comunicar,
            "fraude_confirmada": fraude_confirmada,
        }
    )

    # Inserir valores ausentes intencionais
    for coluna in [
        "idade_segurado",
        "valor_sinistro",
        "tipo_ocorrencia",
        "canal_aviso",
    ]:
        indices = rng.choice(
            df.index,
            size=int(n_sinistros * 0.03),
            replace=False,
        )
        df.loc[indices, coluna] = np.nan

    # Categorias inconsistentes
    indices = rng.choice(
        df.index, size=int(n_sinistros * 0.01), replace=False
    )
    df.loc[indices, "tipo_seguro"] = "automóvel "

    # Idades e valores inválidos
    indices_idade = rng.choice(
        df.index, size=40, replace=False
    )
    df.loc[indices_idade, "idade_segurado"] = -5

    indices_valor = rng.choice(
        df.index, size=40, replace=False
    )
    df.loc[indices_valor, "valor_sinistro"] = -100

    # Duplicatas intencionais
    duplicatas = df.sample(
        n=int(n_sinistros * 0.01),
        random_state=random_state,
    )
    df = pd.concat([df, duplicatas], ignore_index=True)

    return df


def main():
    df = gerar_dataset()

    caminho_saida = RAW_DIR / "sinistros_seguros_sinteticos.csv"
    df.to_csv(caminho_saida, index=False)

    print("=" * 50)
    print("DATASET SINTÉTICO GERADO COM SUCESSO")
    print("=" * 50)
    print(f"Arquivo: {caminho_saida}")
    print(f"Linhas: {len(df):,}")
    print(f"Colunas: {len(df.columns)}")
    print(f"Taxa de fraude simulada: {df['fraude_confirmada'].mean():.2%}")
    print(f"Duplicatas exatas: {df.duplicated().sum():,}")
    print("\nValores ausentes por coluna:")
    print(df.isna().sum().to_string())
    print("\nPrimeiros cinco registros:")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()



