# Resiliência de Ativos Brasileiros em Crises

Projeto público de análise quantitativa de como diferentes classes de ativos
brasileiros se comportaram em três episódios de stress relevantes (2020–2023).

---

## Estrutura do projeto

```
Asset-resilience-br/
├── data/
│   ├── raw/            # Arquivos brutos (Anbima — download manual se necessário)
│   └── processed/      # Dados intermediários processados
├── figures/
│   └── post1/          # Figuras geradas pelo notebook
├── notebooks/
│   └── post1_manifesto.ipynb   # Notebook principal — end-to-end
├── src/
│   ├── data_fetch/
│   │   ├── anbima_fetcher.py   # IFMM e IMA-B 5 (ANBIMA)
│   │   ├── bcb_fetcher.py      # CDI (BCB SGS série 12)
│   │   └── yfinance_fetcher.py # Ibovespa (^BVSP)
│   ├── metrics/
│   │   └── resilience_metrics.py   # Drawdown, recuperação, vol, drawdown normalizado
│   └── viz/
│       └── post1_charts.py         # 4 figuras do post
└── tests/
    └── test_metrics.py         # Testes unitários das métricas
```

---

## Instalação

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows

# Instalar dependências
pip install -r requirements.txt

# Executar testes das métricas
python -m pytest tests/ -v

# Abrir o notebook
jupyter notebook notebooks/post1_manifesto.ipynb
```

---

## Dados utilizados

| Ativo | Fonte | Ticker / Série | Papel na análise |
|-------|-------|----------------|-----------------|
| **Ibovespa** | Yahoo Finance via yfinance | `^BVSP` | Ativo de renda variável |
| **IFMM** | ANBIMA | Índice de Fundos Multimercado | Proxy de multimercado |
| **IMA-B 5** | ANBIMA | NTN-Bs até 5 anos | Renda fixa inflação (curta duration) |
| **CDI** | BCB SGS série 12 | Taxa diária % a.d. | **Referência** de custo de oportunidade |

> **CDI não é um ativo comparável.** Aparece apenas como linha tracejada de referência
> nos gráficos — representa o retorno "sem risco" do mercado interbancário.

### Download dos dados ANBIMA (fallback manual)

O download automático tenta URLs conhecidas com headers de browser. Se falhar:

**IMA-B 5:**
1. Acesse: https://www.anbima.com.br/pt_br/informar/estatisticas/precos-e-indices/ima/ima.htm
2. Procure "Séries Históricas" e baixe o arquivo Excel.
3. Salve como: `data/raw/ima_geral.xlsx`

**IFMM:**
1. Acesse: https://www.anbima.com.br/pt_br/informar/estatisticas/fundos-de-investimento/ifmm.htm
2. Procure o link de séries históricas.
3. Salve como: `data/raw/ifmm.xlsx`

O parser detecta automaticamente o cabeçalho e as colunas relevantes.

---

## Decisões Metodológicas

### Por que essas datas de evento?

**Covid-19 — 21/02/2020**
Sexta-feira de ruptura global após a explosão de casos na Itália e no Irã. Considerada
o primeiro pico real de stress sistêmico internacional — ponto em que o mercado processou
que o vírus havia ultrapassado a contenção. Datas anteriores (ex: 24/01) já mostravam
preocupação, mas 21/02 representa a virada qualitativa de intensidade do stress.

**Invasão da Ucrânia — 24/02/2022**
Data do início da invasão militar russa. Sem ambiguidade como ponto de materialização
do risco geopolítico. Datas anteriores de escalada diplomática (ex: 21/02 — reconhecimento
das repúblicas) são candidatas alternativas, mas 24/02 é o ponto de ruptura definitivo
para os mercados.

**Americanas — 11/01/2023**
Data do fato relevante que comunicou inconsistências contábeis de aproximadamente R$ 20 bilhões.
Evento de crédito privado com impacto imediato em fundos com exposição ao papel. A precisão
da data é máxima — trata-se de divulgação regulatória datada.

### Por que essas janelas?

- **T−5 a T+90 (curta):** captura o impacto imediato e a fase inicial de recuperação.
  Suficiente para eventos com recuperação rápida.
- **T−5 a T+180 (longa):** captura recuperações tardias e permite identificar ativos
  que atingiram novos patamares vs. aqueles que seguiram em tendência negativa.
- **Ambas calculadas:** a comparação entre janelas sinaliza recuperações incompletas
  (drawdown piora entre 90d e 180d) ou consolidação (melhora).
- **T−5 como início:** permite capturar movimentos antecipados nos dias pré-evento.
  O valor em T−5 é a base 100 da normalização.

### Por que esses ativos?

**Ibovespa:** índice de referência do mercado acionário brasileiro. Máxima liquidez
e disponibilidade de dados. Serve como âncora de risco na análise.

**IFMM (Índice de Fundos Multimercado):** representa o desempenho médio ponderado por PL
de fundos multimercado livres — a classe de ativo de gestão ativa mais relevante do mercado
brasileiro institucional. Escolhido em vez de um portfólio sintético 60/40 para preservar
rigor: simula o que um investidor em fundos multimercado de fato obteve.

**IMA-B 5:** índice de NTN-Bs com duration até 5 anos. Representa renda fixa
pós-fixada em IPCA de menor risco de mercado — frequentemente recomendada como âncora
defensiva. A escolha do IMA-B 5 (e não IMA-B total) reduz o ruído de duration longa.

**CDI — por que é referência, não ativo comparável:**
O CDI remunera o custo do dinheiro no overnight interbancário. Por construção, não
sofre drawdowns relevantes — sua "resiliência" seria trivialmente perfeita. Incluí-lo
como ativo distorceria a análise. Sua função é servir de linha de base: se um ativo
não superou o CDI acumulado no período, houve custo de oportunidade real.

### Definição operacional de resiliência

> **Resiliência = preservação de valor + recuperação em prazo razoável.**

Formalmente:
- **Baixo drawdown máximo** na janela de stress.
- **Recuperação ao nível pré-evento** (T−5) em número razoável de dias úteis.

Um ativo com drawdown zero mas sem recuperação não existe nesta análise (se não caiu,
não precisa recuperar). Um ativo que cai 5% e recupera em 10 dias úteis é mais resiliente
que um que cai 3% e não recupera em 180 dias — apesar do drawdown menor.

O **drawdown normalizado** (DD / vol pré-evento) permite comparar a severidade relativa
entre ativos com volatilidades estruturalmente diferentes: um drawdown de -15% em um
ativo com vol anual de 30% é menos severo do que o mesmo -15% em um ativo com vol de 8%.

---

## Executar os testes

```bash
# Testes unitários das métricas
python -m pytest tests/test_metrics.py -v

# Testes inline (sem pytest)
python src/metrics/resilience_metrics.py
python tests/test_metrics.py
```

---

## Linguagem cautelosa — convenção do projeto

Este projeto evita afirmações causais. Expressões como *"o evento causou a queda"*
são substituídas por *"a queda coincidiu com o evento"* ou *"observou-se queda de X%"*.
Isso reflete a impossibilidade de atribuir causalidade a partir de observações de série
temporal com N=3 eventos.

---

## Licença

MIT — ver `LICENSE` para detalhes.
