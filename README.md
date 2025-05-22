# Relatório Técnico da Simulação de Plasma Tokamak

## 1. Introdução

Este relatório detalha a análise da simulação de plasma em tokamaks desenvolvida, com foco na sua validade física e numérica, e estabelece uma comparação com abordagens e referências consagradas na literatura científica e técnica. O objetivo principal da simulação é fornecer uma ferramenta capaz de modelar o equilíbrio do plasma, através da solução da equação de Grad-Shafranov, e a dinâmica temporal das correntes nos circuitos poloidais e no plasma, considerando um modelo de circuito 0D acoplado.

A simulação é composta por dois módulos principais: um solver de equilíbrio Grad-Shafranov (`grad_shafranov_solver.py`) e um simulador dinâmico do tokamak (`tokamak_simulation.py`). O primeiro calcula o equilíbrio MHD (Magneto-Hidrodinâmico) do plasma para um dado conjunto de perfis de pressão e corrente, enquanto o segundo resolve as equações de circuito para as bobinas e o plasma, acoplando-as com os parâmetros de equilíbrio que evoluem no tempo.

O escopo deste relatório abrange a descrição da metodologia implementada em cada módulo, uma análise crítica da validade dos modelos físicos e dos métodos numéricos empregados, e uma discussão sobre como a simulação se posiciona em relação a códigos e técnicas de referência na área de física de plasmas e fusão nuclear.



## 2. Metodologia da Simulação Implementada

A simulação desenvolvida é composta por dois scripts Python principais que interagem para modelar o comportamento do plasma em um tokamak: o `grad_shafranov_solver.py`, responsável por calcular o equilíbrio MHD bidimensional do plasma, e o `tokamak_simulation.py`, que simula a dinâmica temporal das correntes nos circuitos externos e no plasma, acoplada à evolução dos parâmetros de equilíbrio.

### 2.1. Solver de Equilíbrio Grad-Shafranov (`grad_shafranov_solver.py`)

O cálculo do equilíbrio do plasma é fundamental para qualquer análise de estabilidade, transporte ou controle em tokamaks. Este módulo resolve a equação de Grad-Shafranov, uma equação diferencial parcial elíptica não linear que descreve o equilíbrio estático de um plasma toroidal axisimétrico sob a ideal MHD. A equação é dada por:

`Δ* ψ = -μ₀R² dp(ψ)/dψ - F(ψ)dF(ψ)/dψ`

Onde:
*   `ψ(R,Z)` é o fluxo magnético poloidal por radiano.
*   `Δ* = R ∂/∂R (1/R ∂/∂R) + ∂²/∂Z²` é o operador diferencial de Grad-Shafranov em coordenadas cilíndricas (R,Z).
*   `μ₀` é a permeabilidade magnética do vácuo.
*   `R` é a coordenada radial maior.
*   `p(ψ)` é o perfil de pressão do plasma, uma função do fluxo magnético poloidal.
*   `F(ψ) = RB_φ` é o perfil de corrente poloidal (relacionado ao campo magnético toroidal `B_φ`), também uma função do fluxo magnético poloidal.

A solução `ψ(R,Z)` define as superfícies de fluxo magnético concêntricas que caracterizam o equilíbrio do plasma.

**Implementação Numérica:**
O solver implementado utiliza um método de diferenças finitas para discretizar a equação de Grad-Shafranov em uma malha cartesiana regular em (R,Z). A solução é obtida iterativamente através do método de Sobre-Relaxação Sucessiva (SOR). Este método é uma variante do método de Gauss-Seidel que introduz um parâmetro de relaxação `ω` para acelerar a convergência. O solver inclui uma lógica para estimar dinamicamente um valor ótimo para `ω` com base no raio espectral do método de Jacobi (`rho_J_estimated`), uma técnica comum para otimizar a convergência em solvers iterativos. Códigos como o J-Solver também utilizam métodos iterativos para resolver a equação discretizada, embora possam empregar diferentes esquemas de relaxação ou atualização da malha [Referência: J-Solver User Guide].

**Modelos de Perfis de Pressão e Corrente:**
As funções livres `dp(ψ)/dψ` (ou `p(ψ)`) e `F(ψ)dF(ψ)/dψ` (ou `FF′(ψ)`) devem ser especificadas para fechar o sistema. O solver atual permite duas formas funcionais para o perfil de pressão (controlado por `PRESSURE_PROFILE_TYPE`):
1.  **Polinomial:** `dp/dψ_norm = -P₀′ (ψ_norm)^α_p` e `FF′ = FF₀′ (ψ_norm)^α_F`, onde `ψ_norm` é o fluxo normalizado. Esta forma é flexível e comumente usada em códigos de equilíbrio, como mencionado em discussões sobre o CHEASE [Referência: Lutjens et al., CHEASE CPC96].
2.  **Tangente Hiperbólica (tanh):** `p(ψ_norm) = P₀_tanh * (1 - tanh(α_tanh(ψ_norm - ψ_ped))) / (normalização)`. Este perfil permite a formação de um pedestal de pressão, característico de regimes de alto confinamento (H-mode) em tokamaks.

**Escalonamento de P₀′ e FF₀′ com a Corrente de Plasma (Ip):**
Para tornar os perfis mais realistas em diferentes regimes de operação, o solver implementa um escalonamento físico dos termos fonte `P₀′` (amplitude da derivada da pressão) e `FF₀′` (amplitude da derivada de F²/2) com a corrente de plasma alvo (`Ip_target`). Especificamente, `P₀′` é escalado com `(Ip_target / Ip_BASE)²` e `FF₀′` com `(Ip_target / Ip_BASE)`. Esta abordagem reconhece que a pressão e o campo magnético (e, portanto, seus gradientes em relação a ψ) tendem a aumentar com a corrente de plasma. O método `solve()` do solver aceita `Ip_target` e ajusta dinamicamente os parâmetros base (`P0_TANH_BASE`, `P0_PRIME_POLY_GS_BASE`, `FF0_PRIME_POLY_GS_BASE`) antes de iniciar as iterações.

**Extração de Parâmetros Geométricos e Físicos:**
Após a convergência da solução `ψ(R,Z)`, o solver extrai diversos parâmetros importantes:
*   **Geométricos:** Raio maior do eixo magnético (R₀), raio menor (a), elongação (κ) e triangularidade (δ). Estes são determinados identificando a Última Superfície Magnética Fechada (LCFS), geralmente o contorno onde `ψ = ψ_boundary`, utilizando a função `find_contours` da biblioteca `scikit-image`. Um método de fallback é implementado caso um contorno LCFS claro não seja encontrado.
*   **Físicos:** Beta poloidal (βₚ) e indutância interna normalizada (li). Estes são calculados através de integrais sobre o volume do plasma, utilizando o método de Simpson (`scipy.integrate.simpson`) para maior precisão. A corrente de plasma total (Ip_calc) também é calculada integrando a densidade de corrente toroidal `J_φ = R dp/dψ + (1/μ₀R) FF′` sobre a seção transversal do plasma.

**Condições de Contorno e Inicialização:**
O solver opera com condições de contorno de Dirichlet fixas, onde o valor de `ψ` é especificado na borda do domínio computacional retangular (`PSI_BOUNDARY_GS`, tipicamente zero). A inicialização de `ψ` é feita com um perfil analítico aproximado baseado em uma estimativa da posição do eixo magnético e do raio menor, com o valor no eixo magnético tendendo a `PSI_AXIS_TARGET_GS`.



### 2.2. Simulador Dinâmico do Tokamak (`tokamak_simulation.py`)

Este módulo simula a evolução temporal das correntes nos circuitos das bobinas poloidais (PF), do solenóide ôhmico (OH) e no próprio plasma. Ele resolve um sistema de equações diferenciais ordinárias (ODEs) que descrevem a dinâmica do circuito acoplado, levando em consideração a evolução dos parâmetros do plasma, que são periodicamente atualizados pelo solver Grad-Shafranov.

**Modelo de Circuito (0D):**
A simulação emprega um modelo de circuito 0D, onde cada bobina PF, a bobina OH e o plasma são representados como circuitos R-L acoplados. As equações de circuito são da forma:

`d/dt (L I) + R I = V`

Onde:
*   `L` é a matriz de indutâncias (autoindutâncias na diagonal e indutâncias mútuas fora da diagonal).
*   `I` é o vetor das correntes nos circuitos.
*   `R` é a matriz de resistências (geralmente diagonal, com as resistências de cada circuito).
*   `V` é o vetor das tensões aplicadas aos circuitos das bobinas (a tensão no circuito do plasma é tipicamente zero, a menos que se modele explicitamente uma fonte de corrente não indutiva).

Expandindo o termo `d/dt (L I)`, obtemos:

`L dI/dt + (dL/dt) I + R I = V`

O termo `dL/dt` surge porque a matriz de indutâncias `L` não é constante; ela depende da geometria do plasma (R₀, a, κ, δ, li), que evolui no tempo. Assim, `dL/dt = Σ (∂L/∂ξ_k) (dξ_k/dt)`, onde `ξ_k` são os parâmetros do plasma.

**Implementação da Matriz de Indutâncias Física:**
A matriz de indutâncias `L` é calculada utilizando fórmulas semi-analíticas baseadas na geometria das bobinas e do plasma:
*   **Autoindutância de Bobinas Circulares:** Utiliza-se a fórmula padrão `L_coil = μ₀ * N² * R_coil * (ln(8R_coil/r_wire) - 1.75)`, onde `N` é o número de espiras, `R_coil` o raio maior da bobina e `r_wire` o raio do fio.
*   **Indutância Mútua entre Bobinas Circulares Coaxiais:** Calculada usando a fórmula de Maxwell, que envolve integrais elípticas completas de primeiro e segundo tipo (implementadas via `scipy.special.ellipk` e `scipy.special.ellipe`).
*   **Autoindutância do Plasma (Lₚ):** Utiliza-se uma fórmula analítica aproximada que depende dos parâmetros geométricos do plasma (R₀, a, κ, δ) e da indutância interna (li): `L_p ≈ μ₀ * R₀ * (ln(8R₀/a) + li/2 - 2 + C1(κ-1) + C2δ)`.
*   **Indutância Mútua Bobina-Plasma:** O plasma é modelado como uma única espira filamentar em R₀, Z=0, e a fórmula de Maxwell para bobinas coaxiais é aplicada para calcular o acoplamento com cada bobina externa. Esta é uma aproximação comum em modelos 0D, como os encontrados em códigos como o TSC [Referência: TSC User Manual, Jardin et al.].

As derivadas `∂L/∂ξ_k` são calculadas analiticamente para `Lₚ` e aproximadas como zero para as indutâncias mútuas bobina-plasma (exceto para `∂M/∂R₀`, que poderia ser incluída, mas é complexa). Esta aproximação é uma simplificação, mas o impacto principal de `dL/dt` geralmente vem da variação da autoindutância do plasma.

**Modelo de Resistência do Plasma (Rp):**
A resistência do plasma `Rp` é calculada dinamicamente usando a resistividade de Spitzer, que depende da temperatura eletrônica (Te) e da carga iônica efetiva (Z_eff):

`η_spitzer = C_spitzer * Z_eff * lnΛ / Te^(3/2)`

A temperatura eletrônica `Te` é estimada a partir do beta poloidal (βₚ) e da densidade eletrônica média (`n_e`), onde βₚ é fornecido pelo solver Grad-Shafranov. `Rp = η_spitzer * (2πR₀) / (πa²κ)`. Esta abordagem introduz um acoplamento físico importante entre o equilíbrio do plasma e sua resposta resistiva.

**Evolução Dinâmica dos Parâmetros do Plasma (ξ(t)):**
Os parâmetros geométricos e físicos do plasma (R₀, a, κ, δ, li, βₚ), agrupados no vetor `ξ`, não são estáticos. Eles evoluem dinamicamente, relaxando em direção aos valores de equilíbrio (`ξ_GS`) calculados periodicamente pelo solver Grad-Shafranov. Esta relaxação é modelada por uma equação da forma:

`dξ/dt = (ξ_GS - ξ) / τ_ξ`

Onde `τ_ξ` é uma constante de tempo de relaxação para o equilíbrio. Este acoplamento entre a simulação de circuito 0D e o solver de equilíbrio 2D é uma característica crucial de códigos de simulação mais completos como o TSC, permitindo modelar a resposta do plasma a mudanças nas correntes das bobinas ou em fontes de aquecimento.

**Integração Temporal das ODEs:**
O sistema acoplado de ODEs para as correntes `I(t)` e os parâmetros do plasma `ξ(t)` é resolvido utilizando o integrador `solve_ivp` da biblioteca `SciPy`, com o método `BDF` (Backward Differentiation Formulas). Este método é implícito e adequado para sistemas de equações "stiff", que são comuns em simulações de plasma devido às diferentes escalas de tempo envolvidas.



## 3. Análise de Validade da Simulação

A validade da simulação desenvolvida pode ser analisada sob duas perspectivas principais: a validade física dos modelos empregados e a validade numérica dos algoritmos de solução.

### 3.1. Validade Física

**Solver Grad-Shafranov:**
*   **Equilíbrio MHD Ideal:** A equação de Grad-Shafranov é derivada da MHD ideal, que assume um plasma perfeitamente condutor e em equilíbrio estático. Esta é uma aproximação fundamental e amplamente utilizada para descrever o estado base de plasmas em tokamaks [Referência: Wesson, J. Tokamaks]. A simulação adere a esta premissa.
*   **Perfis de Pressão e Corrente:** A escolha dos perfis `p(ψ)` e `F(ψ)` é crucial. Os perfis polinomiais e do tipo tangente hiperbólica implementados são formas funcionais comumente usadas na modelagem de equilíbrio. Por exemplo, o código CHEASE utiliza parametrizações semelhantes para `dp/dψ` e `FF′` baseadas em potências de `ψ_norm` [Referência: Lutjens et al., CHEASE CPC96]. A capacidade de modelar um pedestal de pressão com o perfil `tanh` é fisicamente relevante para simular regimes H-mode. A flexibilidade na escolha destes perfis é um ponto positivo, mas a determinação de perfis que correspondam realisticamente a um cenário experimental específico requereria validação com dados diagnósticos, como faz o código EFIT [Referência: Lao, L. L., et al. Nuclear Fusion 25.11 (1985): 1611].
*   **Escalonamento de P₀′ e FF₀′:** O escalonamento implementado (`P₀′ ∝ Ip²`, `FF₀′ ∝ Ip`) é uma tentativa de capturar a dependência da pressão e do campo poloidal com a corrente de plasma total. Fisicamente, espera-se que plasmas com maior corrente possam confinar maior pressão e gerar campos poloidais mais intensos. Esta abordagem é uma simplificação, pois a relação exata pode depender de outros fatores como o perfil de aquecimento e transporte. No entanto, introduz um grau de realismo maior do que manter P₀′ e FF₀′ constantes para todos os níveis de Ip.

**Simulador Dinâmico:**
*   **Modelo de Circuito 0D:** O modelo de circuito acoplado para as bobinas e o plasma é uma abordagem 0D padrão para simulações de controle e evolução de cenários em tokamaks. Códigos como o TSC (Tokamak Simulation Code) utilizam modelos de circuito semelhantes para descrever a interação eletromagnética entre o plasma e os condutores externos [Referência: Jardin, S. C., et al. Journal of Computational Physics 66.2 (1986): 481-507; TSC User Manual]. A principal limitação do modelo 0D é que ele não resolve a distribuição espacial de corrente e temperatura dentro do plasma, tratando-o como um único condutor com parâmetros globais.
*   **Matriz de Indutâncias:** A utilização de fórmulas semi-analíticas para as autoindutâncias e indutâncias mútuas, baseadas na geometria das bobinas e do plasma (modelado como filamento para acoplamento mútuo), é uma melhoria significativa em relação a matrizes de indutância ad-hoc. Estas fórmulas são derivadas de princípios básicos da eletrodinâmica. A precisão depende da validade das aproximações geométricas (e.g., filamento para o plasma, seção transversal circular para bobinas).
*   **Resistência do Plasma (Spitzer):** A implementação da resistividade de Spitzer com dependência de `Te^(−3/2)` e `Z_eff` é fisicamente correta para um plasma colisional clássico. A estimativa de `Te` a partir de `βₚ` e `n_e` é uma aproximação razoável para um modelo 0D, pois `βₚ` relaciona a energia cinética do plasma com a energia do campo magnético. No entanto, em plasmas de tokamak reais, o transporte anômalo frequentemente domina a resistividade efetiva, especialmente em regiões mais frias ou periféricas.
*   **Evolução de ξ(t):** O modelo de relaxação dos parâmetros de equilíbrio (`ξ`) em direção aos valores calculados pelo solver GS (`ξ_GS`) com uma constante de tempo `τ_ξ` é uma forma de acoplar a evolução lenta do equilíbrio MHD com a dinâmica mais rápida do circuito. O valor de `τ_ξ` deve representar escalas de tempo físicas associadas ao transporte de energia e partículas ou à redistribuição de corrente, que são tipicamente mais lentas que a escala de tempo L/R dos circuitos. Códigos como o DINA seguem abordagens semelhantes para o acoplamento entre equilíbrio e evolução temporal [Referência: Khayrutdinov, R. R., and V. E. Lukash. Journal of Computational Physics 109.2 (1993): 193-201].

### 3.2. Validade Numérica

**Solver Grad-Shafranov:**
*   **Método de Diferenças Finitas e SOR:** A discretização por diferenças finitas em uma malha cartesiana é um método robusto e relativamente simples de implementar. O método SOR é eficaz para resolver o sistema linear resultante em cada iteração (se a não linearidade dos perfis for tratada com Picard, ou para a própria equação se os termos fonte forem fixos). A convergência depende da escolha de `ω` e da dominância diagonal da matriz discretizada. A estimativa adaptativa de `ω` implementada visa otimizar essa convergência. A resolução da malha (`NR_GS`, `NZ_GS`) impacta diretamente a precisão da solução e a capacidade de resolver gradientes íngremes.
*   **Convergência:** O critério de convergência baseado no erro RMS da solução `ψ` entre iterações é um padrão. A robustez do solver foi testada, e ele converge para uma ampla gama de parâmetros, especialmente após o ajuste dos valores base de `P₀′` e `FF₀′` para níveis fisicamente razoáveis e a implementação do escalonamento.
*   **Extração de Parâmetros:** A utilização de `skimage.measure.find_contours` para o LCFS e `scipy.integrate.simpson` para integrais melhora a precisão na extração de parâmetros em comparação com métodos mais simples. O fallback para a geometria em caso de não detecção do LCFS garante robustez.

**Simulador Dinâmico:**
*   **Integrador ODE (BDF):** O método BDF (Backward Differentiation Formulas) é uma boa escolha para sistemas de ODEs que podem ser "stiff", como é o caso aqui devido às múltiplas escalas de tempo (L/R das bobinas, `τ_ξ`, etc.). A `solve_ivp` da SciPy com BDF é uma ferramenta robusta e bem testada.
*   **Estabilidade:** A estabilidade da simulação acoplada depende da consistência entre as escalas de tempo e da robustez das atualizações do solver GS. Se o solver GS falhar em convergir ou se os parâmetros do plasma mudarem muito drasticamente entre as chamadas ao GS, isso pode levar a instabilidades numéricas na simulação dinâmica. O intervalo de chamada ao GS (`gs_config["call_interval"]`) e a constante de tempo `τ_ξ` são parâmetros cruciais para a estabilidade e precisão do acoplamento.
*   **Cálculo de `dL/dt`:** A aproximação de que as derivadas das indutâncias mútuas bobina-plasma em relação a `a, κ, δ, li` são negligenciáveis simplifica o cálculo de `dL/dt`. Embora a variação de `Lₚ` com esses parâmetros seja geralmente dominante, em cenários com grandes variações de forma, essa aproximação pode introduzir alguma imprecisão.

### 3.3. Comparação Qualitativa com Resultados Esperados

Os resultados das simulações (não mostrados em detalhe aqui, mas observados durante o desenvolvimento) demonstram comportamentos qualitativamente esperados:
*   A corrente de plasma aumenta com a aplicação de tensão no solenóide OH.
*   Os parâmetros geométricos do plasma (R₀, a, etc.) evoluem em resposta às mudanças nas correntes das bobinas PF e do plasma, tendendo aos novos equilíbrios calculados pelo solver GS.
*   A temperatura do plasma (inferida de βₚ) e, consequentemente, a resistência do plasma, respondem às mudanças na corrente e no equilíbrio.

**Limitações do Modelo Atual:**
*   **Modelo 0D para Dinâmica:** Não captura perfis espaciais de corrente, temperatura ou densidade dentro do plasma, nem fenômenos de transporte 1D/2D.
*   **Resistência do Plasma:** Apenas Spitzer clássica; não inclui efeitos neoclássicos ou anômalos que podem ser significativos.
*   **Fonte de Partículas e Energia:** Não há modelos explícitos para fontes de partículas (gás puffing, pellets) ou aquecimento externo (NBI, RF), embora o escalonamento de P₀′ possa implicitamente representar um nível de aquecimento.
*   **Estabilidade MHD:** O solver GS calcula equilíbrios, mas a simulação dinâmica não avalia a estabilidade desses equilíbrios a modos MHD (e.g., kinks, VDEs), embora a evolução de `ξ` possa ser instável se os equilíbrios se tornarem inacessíveis.
*   **Interação Plasma-Parede:** Simplificada pela condição de contorno fixa do solver GS e pela ausência de um modelo de reciclagem ou impurezas.



## 4. Referências e Comparação com a Literatura

A simulação desenvolvida, embora simplificada em comparação com grandes códigos de produção, incorpora muitos dos princípios e técnicas encontrados na literatura e em ferramentas de simulação de tokamaks consagradas.

### 4.1. Solvers de Equilíbrio Grad-Shafranov

O solver Grad-Shafranov implementado utiliza um método de diferenças finitas e SOR, que é uma abordagem clássica. Códigos mais avançados como:
*   **CHEASE (Cubic Hermite Element Axisymmetric Static Equilibrium):** Utiliza elementos finitos Hermite bicúbicos, que oferecem maior precisão, especialmente para o cálculo de derivadas do fluxo magnético (campos magnéticos). A parametrização dos perfis de pressão e corrente em CHEASE, usando funções de `ψ_norm`, é conceitualmente similar à abordagem do solver desenvolvido [Referência: Lutjens, H., Bondeson, A., & Sauter, O. (1996). The CHEASE code for toroidal MHD equilibria. Computer Physics Communications, 97(3), 219-260].
*   **EFIT (Equilibrium Fitting):** É amplamente utilizado para reconstruir o equilíbrio do plasma a partir de dados diagnósticos experimentais. Ele também resolve a equação de Grad-Shafranov, ajustando os perfis de `p(ψ)` e `F(ψ)` para corresponder às medições. A capacidade do nosso solver de usar diferentes formas funcionais para os perfis (polinomial, tanh) e de escalar os termos fonte com `Ip` são passos iniciais na direção da flexibilidade necessária para tal ajuste, embora a reconstrução a partir de dados não esteja implementada [Referência: Lao, L. L., St. John, H., Stambaugh, R. D., Kellman, A. G., & Pfeiffer, W. (1985). Reconstruction of current profile parameters and plasma shapes in tokamaks. Nuclear Fusion, 25(11), 1611].
*   **J-Solver:** Um código de equilíbrio de contorno fixo que também resolve a equação de Grad-Shafranov iterativamente, com foco na especificação do perfil de corrente paralela `J_paralela(ψ)` [Referência: Menard, J. E., Jardin, S. C., Kaye, S. M., Kessel, C. E., & Manickam, J. (2000). J-Solver Equilibrium Code User Guide. Princeton Plasma Physics Laboratory]. A escolha das funções livres (p′, FF′ vs. p′, J_paralela) é uma diferença fundamental na formulação, mas o objetivo de encontrar um equilíbrio autoconsistente é o mesmo.

A principal diferença do solver atual para esses códigos de referência reside na sofisticação do método numérico (diferenças finitas vs. elementos finitos, métodos de malha adaptativa ausentes) e na capacidade de lidar com contornos de plasma livres e reconstrução a partir de dados experimentais.

### 4.2. Modelos de Simulação Dinâmica de Tokamaks (0D/1D)

O simulador dinâmico implementado é um modelo 0D que captura a evolução temporal das correntes de circuito e dos parâmetros globais do plasma. Esta abordagem é comum para estudos de controle e desenvolvimento de cenários.
*   **TSC (Tokamak Simulation Code):** É um código de referência para simulações de evolução temporal em tokamaks, incluindo a dinâmica do plasma deformável, controle de forma e posição, e aquecimento. O TSC resolve equações de circuito para os condutores externos e equações de transporte 1.5D para o plasma, acopladas a um solver de equilíbrio Grad-Shafranov para o plasma 2D. O modelo de circuito e o acoplamento com um solver GS no nosso simulador são conceitualmente alinhados com a abordagem do TSC, embora o TSC seja significativamente mais complexo, incluindo modelos de transporte detalhados e a capacidade de simular eventos dinâmicos como VDEs (Vertical Displacement Events) [Referência: Jardin, S. C., Pomphrey, N., & DeLucia, J. (1986). Dynamic modeling of transport and positional control of tokamaks. Journal of Computational Physics, 66(2), 481-507; TSC User Manual, PPPL].
*   **DINA (Dynamics of Non-circular Axisymmetric Plasmas):** Similar ao TSC, o DINA é outro código proeminente para simulação da dinâmica de plasmas em tokamaks, focando na evolução do equilíbrio e no controle. Ele também emprega um modelo de circuito para os condutores e resolve a equação de Grad-Shafranov para o equilíbrio do plasma, acoplando-os de forma autoconsistente [Referência: Khayrutdinov, R. R., & Lukash, V. E. (1993). DINA: A code for the simulation of the dynamics of a noncircular axisymmetric plasma in a tokamak. Journal of Computational Physics, 109(2), 193-201].
*   **ASTRA (Automated System for Transport Analysis):** Embora primariamente um código de transporte 1.5D, o ASTRA também inclui módulos para equilíbrio e pode ser usado em conjunto com modelos de circuito para simulações preditivas. O foco do ASTRA está mais nos perfis internos do plasma (temperatura, densidade) do que na dinâmica dos circuitos externos, mas a necessidade de um equilíbrio consistente é comum a todos esses códigos.

A simulação atual captura a essência do acoplamento circuito-equilíbrio, mas carece dos modelos de transporte 1D/1.5D e da capacidade de simular a evolução detalhada dos perfis internos do plasma que códigos como TSC, DINA e ASTRA possuem.

### 4.3. Fórmulas de Indutância e Resistência

*   **Indutâncias:** As fórmulas utilizadas para autoindutância de bobinas circulares e indutância mútua entre elas (baseadas em integrais elípticas) são padrões da eletrodinâmica [Referência: Grover, F. W. (1973). Inductance calculations: working formulas and tables. Courier Corporation]. A modelagem da autoindutância do plasma e seu acoplamento mútuo com bobinas externas usando aproximações analíticas ou semi-analíticas é uma prática comum em modelos 0D/1D para evitar o custo computacional de cálculos de campo 2D/3D a cada passo de tempo.
*   **Resistividade de Spitzer:** A fórmula para a resistividade de Spitzer é um resultado clássico da teoria cinética de plasmas colisionais [Referência: Spitzer Jr, L., & Härm, R. (1953). Transport phenomena in a completely ionized gas. Physical Review, 89(5), 977]. Sua aplicação na simulação, com dependência de Te, é fisicamente fundamentada para a componente clássica da resistência do plasma.

## Melhorias Futuras
*   **Solver GS:** Considerar a implementação de elementos finitos ou uma malha adaptativa para maior precisão e flexibilidade geométrica. Implementar a capacidade de lidar com contornos de plasma livres definidos por bobinas externas.
*   **Simulador Dinâmico:** Evoluir para um modelo 1D ou 1.5D, incorporando equações de transporte para perfis de temperatura e densidade. Incluir modelos mais sofisticados para resistência do plasma (neoclássica, anômala) e fontes de aquecimento/corrente.
*   **Acoplamento e Controle:** Implementar laços de controle PID para regulação da corrente de plasma, posição e forma, utilizando as tensões das bobinas como atuadores.
*   **Validação e Verificação:** Comparar sistematicamente os resultados com códigos de referência e, se possível, com dados experimentais para cenários específicos.
*   **Engenharia de Código:** Modularizar ainda mais o código, adicionar testes unitários abrangentes e externalizar parâmetros de simulação para arquivos de configuração.

Apesar das limitações inerentes a um modelo desta complexidade desenvolvido em um escopo limitado, a simulação atual serve como uma excelente plataforma educacional e de pesquisa inicial, permitindo a exploração de muitos aspectos importantes da física e da operação de tokamaks. As melhorias sugeridas podem transformá-la progressivamente em uma ferramenta de simulação mais poderosa e preditiva.

## 6. Lista de Referências Bibliográficas

*   Grover, F. W. (1973). *Inductance calculations: working formulas and tables*. Courier Corporation.
*   Jardin, S. C., Pomphrey, N., & DeLucia, J. (1986). Dynamic modeling of transport and positional control of tokamaks. *Journal of Computational Physics, 66*(2), 481-507.
*   Khayrutdinov, R. R., & Lukash, V. E. (1993). DINA: A code for the simulation of the dynamics of a noncircular axisymmetric plasma in a tokamak. *Journal of Computational Physics, 109*(2), 193-201.
*   Lao, L. L., St. John, H., Stambaugh, R. D., Kellman, A. G., & Pfeiffer, W. (1985). Reconstruction of current profile parameters and plasma shapes in tokamaks. *Nuclear Fusion, 25*(11), 1611.
*   Lutjens, H., Bondeson, A., & Sauter, O. (1996). The CHEASE code for toroidal MHD equilibria. *Computer Physics Communications, 97*(3), 219-260.
*   Menard, J. E., Jardin, S. C., Kaye, S. M., Kessel, C. E., & Manickam, J. (2000). *J-Solver Equilibrium Code User Guide*. Princeton Plasma Physics Laboratory.
*   Spitzer Jr, L., & Härm, R. (1953). Transport phenomena in a completely ionized gas. *Physical Review, 89*(5), 977.
*   Wesson, J. (2011). *Tokamaks* (4th ed.). Oxford University Press.
*   *TSC User Manual*. Princeton Plasma Physics Laboratory. (Disponível online em w3.pppl.gov/topdac/tscman.pdf)

