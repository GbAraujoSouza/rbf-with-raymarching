# Guia de implementacao: ray marching em campos RBF

## Objetivo

Este guia descreve uma evolucao incremental para transformar o projeto em uma bancada interativa de estudo sobre ray marching em campos RBF.

A ideia central e investigar como estimar um bom passo de marching quando a funcao RBF nao e uma SDF perfeita. A aplicacao deve permitir comparar estrategias de passo, medir comportamento, ajustar parametros no navegador e evoluir para cenas visualmente interessantes com objetos reais e renderizacao mais bonita.

## Visao geral das entregas

1. Organizar o projeto como bancada de experimentos.
2. Adicionar metricas visuais e numericas.
3. Formalizar estrategias de passo.
4. Criar uma interface interativa.
5. Melhorar a construcao da RBF.
6. Adicionar cenas procedurais de validacao.
7. Importar nuvens de pontos reais pequenas.
8. Adicionar renderizacao bonita incremental.
9. Documentar resultados e conclusoes.

## Etapa 1: Organizar o projeto como bancada de experimentos

### Entrega

Uma estrutura onde RBF, cena, parametros de marching, renderizacao e interface estejam separados o suficiente para evoluir sem misturar responsabilidades.

### Como fazer

Crie tipos centrais para representar a configuracao do experimento:

```ts
export interface MarchingConfig {
    strategy: StepStrategy;
    epsilon: number;
    maxDistance: number;
    maxSteps: number;
    correctionLinear: number;
    correctionPower: number;
}

export type StepStrategy =
    | "naive"
    | "corrected"
    | "bounded"
    | "gradient";
```

Mantenha `src/rbf.ts` focado em gerar amostras, resolver pesos e empacotar dados para GPU. Mantenha `src/renderer.ts` focado em WebGPU: device, pipeline, buffers, uniforms e frame loop.

Crie um modulo novo, por exemplo `src/experiment.ts`, para concentrar o estado atual:

```ts
export interface ExperimentState {
    sceneId: string;
    marching: MarchingConfig;
    renderMode: RenderMode;
    showControlPoints: boolean;
}
```

O renderer deve receber esse estado e refletir suas mudancas em uniforms ou buffers. Parametros que nao exigem recalculo da RBF devem apenas atualizar uniforms.

### Criterio de pronto

- A esfera atual continua renderizando.
- `npm run typecheck` passa.
- `npm run build` passa.
- Parametros hoje fixos em `renderer.ts` e no shader passam a vir de uma configuracao explicita.

## Etapa 2: Adicionar metricas visuais e numericas

### Entrega

A aplicacao mostra informacoes suficientes para avaliar o comportamento da estrategia de passo atual antes de introduzir novas variacoes.

### Como fazer

No shader, acompanhe pelo menos:

- numero de passos ate o hit;
- miss;
- saturacao em `MAX_MARCHING_STEPS`;
- distancia final encontrada.

Para comecar, essas metricas podem ser usadas apenas para colorir a imagem. Depois, podem ser agregadas em buffers de estatistica.

Adicione modos de visualizacao:

- superficie sombreada;
- pontos de controle;
- heatmap de quantidade de passos;
- misses e regioes sem convergencia;
- normal estimada.

O heatmap e a visualizacao mais importante para esta etapa. Ele deve deixar claro quais regioes da imagem gastam muitos passos.

No TypeScript, medir tambem:

- FPS;
- tempo medio de frame;
- cena atual;
- estrategia atual;
- quantidade de samples RBF.

### Criterio de pronto

- E possivel ver onde o marching e caro.
- E possivel avaliar a estrategia atual sem depender de impressao subjetiva.
- A UI mostra FPS, cena atual e parametros principais.

## Etapa 3: Formalizar as estrategias de passo

### Entrega

Um conjunto inicial de estrategias de marching comparaveis e selecionaveis pela UI ou pelo estado do experimento.

### Como fazer

Use as metricas da etapa anterior como baseline. Antes de adicionar uma estrategia nova, registre como a estrategia atual se comporta na esfera: heatmap, FPS aproximado, regioes problemáticas e configuracao usada.

No shader WGSL, transforme a escolha do passo em uma funcao unica:

```wgsl
fn estimateStep(point: vec3f, fieldValue: f32, rayDirection: vec3f) -> f32 {
    // Escolhe a estrategia usando um inteiro vindo dos uniforms.
}
```

Implemente estas estrategias iniciais:

### Passo ingenuo

Usa diretamente:

```text
step = abs(f(p))
```

Essa estrategia serve como baseline. Ela tende a funcionar quando o campo se comporta como uma SDF, mas pode falhar ou ser ineficiente quando a RBF tem escala ruim.

### Passo corrigido

Usa a formula empirica atual:

```text
step = correctionLinear * pow(max(abs(f(p)), epsilon), correctionPower)
```

Essa estrategia deve continuar existindo porque ja e o ponto de partida do projeto.

### Passo com bound espacial

Combina a estrategia corrigida com um limite espacial simples, como bounding sphere ou bounding box. A ideia e acelerar regioes longe do objeto e evitar marching desnecessario fora da area util.

Exemplo conceitual:

```text
step = max(correctedStep, distanceToBoundingRegion)
```

### Passo por gradiente local

Estima o gradiente do campo e aproxima o passo por:

```text
step = abs(f(p)) / max(length(grad f(p)), minGradient)
```

Essa e uma das estrategias mais importantes para o estudo, porque tenta corrigir a escala local do campo RBF. Se o campo cresce rapido, o passo diminui. Se o campo cresce devagar, o passo aumenta.

### Criterio de pronto

- A estrategia pode ser alterada sem recompilar o shader.
- Todas as estrategias renderizam a esfera.
- O heatmap permite comparar as estrategias visualmente.
- Existe uma estrategia conservadora para servir como referencia visual.

## Etapa 4: Criar interface interativa

### Entrega

Uma interface no navegador para ajustar parametros da RBF, do marching e da visualizacao.

### Como fazer

Use HTML, CSS e TypeScript puro. Nao e necessario adicionar framework frontend para esta fase.

Adicione controles para:

- cena;
- estrategia de passo;
- modo de visualizacao;
- `epsilon` do marching;
- `maxDistance`;
- `maxSteps`;
- `correctionLinear`;
- `correctionPower`;
- `epsilon` da RBF;
- quantidade de amostras de superficie;
- mostrar ou esconder pontos.

Organize os controles como um painel lateral ou overlay compacto sobre o canvas. O canvas deve continuar sendo o foco da aplicacao.

Ao alterar parametros:

- parametros de shader atualizam uniforms;
- parametros de RBF recalculam pesos e recriam buffers;
- mudancas de camera ou parametros visuais resetam acumulacao temporal futura;
- mudancas grandes devem evitar travar a UI, principalmente se o solve da RBF ficar caro.

### Criterio de pronto

- O usuario consegue explorar parametros sem editar codigo.
- Alteracoes leves nao reinicializam WebGPU inteiro.
- Alteracoes na RBF recriam buffers corretamente.

## Etapa 5: Melhorar a construcao da RBF

### Entrega

Uma RBF mais robusta, com sinal coerente dentro e fora da superficie, adequada para estudar ray marching.

### Como fazer

Reintroduza constraints com offset quando houver normal conhecida:

```text
ponto na superficie: target = 0
ponto externo: target = +offset
ponto interno: target = -offset
```

Para uma esfera procedural, as normais sao triviais: a propria direcao do ponto na esfera.

Mantenha anchors externos para estabilidade, mas torne isso configuravel:

- habilitar/desabilitar anchors;
- distancia dos anchors;
- target dos anchors;
- quantidade ou padrao de anchors.

Adicione regularizacao ao sistema linear:

```text
M[i][i] += regularization
```

Teste variacoes de:

- `rbfEpsilon`;
- offset;
- regularizacao;
- numero de amostras;
- tipo de kernel.

### Criterio de pronto

- A esfera tem sinal coerente dentro e fora.
- O solve nao falha facilmente em configuracoes razoaveis.
- A qualidade visual melhora ou fica mais previsivel ao variar parametros.

## Etapa 6: Adicionar cenas procedurais de validacao

### Entrega

Mais de uma cena simples para testar as estrategias de passo em situacoes diferentes.

### Como fazer

Adicione uma abstracao de cena:

```ts
export interface SceneDefinition {
    id: string;
    label: string;
    createSamples(config: RbfFitConfig): RbfSample[];
    bounds: SceneBounds;
}
```

Comece com:

- esfera;
- duas esferas proximas;
- torus;
- forma assimetrica procedural;
- nuvem sintetica com ruido.

A esfera sozinha pode esconder problemas porque e simetrica e simples. A forma assimetrica e a nuvem com ruido ajudam a revelar falhas de convergencia e instabilidade no passo.

### Criterio de pronto

- A UI permite trocar de cena.
- Cada cena renderiza usando as mesmas estrategias.
- O heatmap mostra diferencas claras entre cenas e estrategias.

## Etapa 7: Importar nuvens de pontos reais pequenas

### Entrega

Suporte inicial a objetos reais usando nuvens de pontos publicas pequenas.

### Como fazer

Comece com arquivos pequenos e pre-processados. Para a v1, prefira um formato interno simples:

```json
{
    "points": [[0, 0, 0]],
    "normals": [[0, 1, 0]]
}
```

Se o dataset vier em `PLY` ou `OBJ`, crie um processo simples de conversao para esse JSON interno. Isso reduz complexidade no browser.

Normalize os dados importados:

- centralizar na origem;
- escalar para caber em uma bounding sphere previsivel;
- limitar a quantidade de pontos;
- remover duplicatas obvias;
- opcionalmente aplicar amostragem aleatoria ou farthest point sampling.

Se nao houver normais:

- comece usando apenas pontos de superficie e anchors;
- depois estime normais por vizinhanca local;
- trate isso como uma etapa de pesquisa separada, porque normais ruins afetam diretamente a qualidade da RBF.

### Criterio de pronto

- Pelo menos uma nuvem real pequena renderiza.
- O dataset carrega rapidamente.
- A comparacao de estrategias continua funcionando.

## Etapa 8: Adicionar renderizacao bonita incremental

### Entrega

Um modo visual mais atraente, sem transformar a primeira versao em um path tracer completo.

### Como fazer

Implemente em fases:

1. Anti-aliasing por jitter.
2. Acumulacao temporal.
3. Ambient occlusion simples.
4. Sombras por ray marching.
5. Materiais simples.

Para acumulacao temporal, mantenha uma textura com o resultado anterior e acumule frames quando a camera e os parametros estiverem parados.

Quando qualquer parametro mudar, resete o contador de acumulacao.

O modo bonito deve ser opcional. O modo de experimento precisa continuar rapido e legivel.

### Criterio de pronto

- O modo bonito melhora claramente a apresentacao.
- A aplicacao continua interativa.
- E possivel alternar entre modo experimento e modo apresentacao.

## Etapa 9: Documentar resultados

### Entrega

Um registro dos aprendizados, comparacoes e decisoes tecnicas.

### Como fazer

Crie um arquivo como `docs/results.md` contendo:

- descricao das estrategias testadas;
- cenas usadas;
- parametros importantes;
- screenshots;
- tabela com passos medios, qualidade e falhas;
- conclusoes parciais.

Uma tabela inicial pode seguir este formato:

| Cena | Estrategia | Passos medios | Falhas | Observacao |
| --- | --- | ---: | ---: | --- |
| Esfera | Ingenua | TBD | TBD | Baseline |
| Esfera | Gradiente | TBD | TBD | Deve reduzir erro de escala local |

### Criterio de pronto

- O projeto explica claramente o que esta sendo investigado.
- As conclusoes nao ficam apenas implícitas no codigo.
- Existe material suficiente para evoluir para artigo, post ou apresentacao.

## Ordem recomendada

1. Organizar estado e configuracao do experimento.
2. Adicionar heatmap e metricas para a estrategia atual.
3. Implementar estrategias de passo selecionaveis.
4. Criar UI de parametros.
5. Melhorar constraints da RBF.
6. Adicionar cenas procedurais.
7. Importar nuvem real pequena.
8. Adicionar renderizacao bonita incremental.
9. Documentar resultados.

## Cuidados tecnicos

- Nao trate a RBF como SDF perfeita sem verificar. Esse e justamente o ponto central do estudo.
- Sempre compare contra uma estrategia conservadora.
- Separe parametros que exigem recalculo da RBF de parametros que apenas mudam o shader.
- Evite datasets grandes no inicio.
- Prefira cenas simples para validar uma hipotese antes de investir em visual complexo.
- Registre configuracoes que funcionam bem e configuracoes que falham.

## Proxima acao sugerida

Comece pela Etapa 1 e Etapa 2. Primeiro organize o estado minimo do experimento, depois adicione heatmap e metricas para entender o comportamento atual antes de criar novas estrategias de passo.
