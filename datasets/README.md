# Datasets

Basicamente esta é a pasta dos datasets. 

Já tem aqui datasets de diferentes links que o sor meteu.

Alguns destes ficheiros têm dados misturados e depois têm labels em cada entrada para dizer se é IA ou Human.

Outros já estão divididos, tendo um ficheiro para apenas um tipo de escrita.

Dei merge a todos os datasets mantendo dois (por causa do tamanho) com apenas as colunas importantes: `text` e `label`. text com o texto em questao e label para saber se foi AI ou Human.

A parte seguinte é dar split para duas pastas, a pasta `train`e a pasta `test`. 75% foi para treino e 25% para teste.