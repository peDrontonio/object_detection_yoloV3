# SAM2 Video Segmentation

Este projeto implementa a segmentação de vídeos utilizando o modelo **SAM2**. O código realiza o processamento de frames (imagens) extraídas de vídeos, aplica segmentação dos objetos de interesse e salva os resultados em formato COCO, com anotações e máscaras sobrepostas.

## Descrição do Projeto

O pipeline de segmentação desenvolvido neste projeto realiza as seguintes etapas:

- **Pré-processamento dos Frames:** Lista e organiza os arquivos de imagem de um vídeo, padronizando seus nomes para manter a sequência correta.
- **Configuração do Ambiente:** Define o dispositivo de computação (GPU com CUDA, MPS em Macs ou CPU) e realiza ajustes de performance utilizando tipos de dados como bfloat16 e TF32.
- **Instanciação do Preditor SAM2:** Carrega o checkpoint e arquivo de configuração para criar um objeto preditor responsável pela segmentação.
- **Processamento das Máscaras:** Utiliza anotações no formato COCO para decodificar e aplicar as máscaras sobre os frames, utilizando técnicas como Run-Length Encoding (RLE).
- **Propagação da Segmentação:** Propaga a segmentação para todos os frames do vídeo, armazenando os resultados em um dicionário.
- **Visualização e Salvamento:** Exibe os resultados para verificação e, em seguida, salva os frames com sobreposição das máscaras, as imagens das máscaras isoladas e um arquivo JSON com as anotações no padrão COCO.

## Requisitos

- **Python 3.x**
- **Bibliotecas:**
  - PyTorch
  - OpenCV
  - NumPy
  - Matplotlib
  - Pillow (PIL)
  - pycocotools

Verifique as dependências no arquivo `requirements.txt` (caso exista) ou instale manualmente com o gerenciador de pacotes `pip`:

```bash 
conda create --name object_detection_amb python=3.10  # specify the Python version you need
conda activate object_detection_amb
pip install -r requirements.txt
```
## Estrutura do Projeto

- **run_sam2.py:** Script principal que executa o pipeline de segmentação utilizando o SAM2.

- **Aula0(EXTRA).ipynb:** Notebook que contém uma explicação detalhada, célula a célula, do código utilizado.

    Diretórios com dados e resultados:
    ```bash
        /home/nexus/sam2/sam2/datasets/giovanna/annotations/instances_default.json: Arquivo com anotações no formato COCO.

        /home/nexus/sam2/sam2/datasets/giovanna/images/default: Diretório com os frames do vídeo.
    Diretórios de output para salvar os frames processados, máscaras e o arquivo JSON com as anotações.
    ```    
        
## Como Utilizar
   
- **Configure os Caminhos:**
    Verifique e atualize os caminhos para os arquivos de anotações, diretório dos frames, checkpoint do modelo e arquivo de configuração no script run_sam2.py, conforme seu ambiente de execução.

- **Execução do Script:**
    Execute o script principal utilizando o Python:

```bash 
python run_sam2.py
``` 

    Durante a execução, o código exibirá informações sobre a disponibilidade da GPU e exibirá os frames com as máscaras para verificação.

    Verifique os Resultados:
    Após o processamento, os seguintes resultados serão gerados:

        Frames com as máscaras sobrepostas salvos no diretório de output.

        Imagens das máscaras isoladas.

        Arquivo JSON contendo as anotações formatadas segundo o padrão COCO.

## Funcionamento do Código

- **O pipeline está organizado em diversas etapas, conforme descrito a seguir:**

- **Importação de Bibliotecas e Configuração do Ambiente:**
    São importadas bibliotecas essenciais para manipulação de imagens, operações numéricas e computação em GPU. Também é feita a configuração de variáveis de ambiente para garantir a compatibilidade com dispositivos MPS em Macs, por exemplo.

- **Carregamento e Preparação dos Frames:**
    O código lista os arquivos de imagem no diretório e os ordena, padronizando os nomes (ex.: adicionando zeros à esquerda) para garantir a sequência correta.

- **Seleção do Dispositivo de Computação e Ajustes de Performance:**
    Verifica se há uma GPU compatível com CUDA ou suporte para MPS e, de acordo com o dispositivo disponível, ajusta as configurações como o uso de bfloat16 e TF32 para melhorar a performance.

- **Instanciação do Preditor SAM2:**
    Com base nos arquivos de configuração e checkpoint, é criado o objeto predictor que realiza a segmentação de vídeo.

- **Processamento das Máscaras com Anotações COCO:**
    Utiliza-se o arquivo de anotações para decodificar as máscaras (em formato RLE) e aplicar a segmentação ao utilizar o método add_new_mask do preditor.

- **Propagação e Visualização:**
    As máscaras são propagadas ao longo dos frames do vídeo e armazenadas em um dicionário para visualização. Funções auxiliares (como show_mask, show_points e show_box) ajudam a exibir os resultados.

- **Salvamento dos Resultados:**
    São gerados os frames com sobreposição das máscaras, as máscaras isoladas e um arquivo JSON com as anotações, salvos nos respectivos diretórios de output.

## Notas e Considerações

- **Performance:**
    A utilização de uma GPU com suporte a CUDA é recomendada para acelerar o processamento. Caso o dispositivo utilize CPU, o processamento poderá ser significativamente mais lento.

- **Suporte MPS:**
    Em dispositivos Apple, embora o suporte MPS esteja disponível, ele é ainda preliminar e os resultados podem apresentar pequenas variações.

- **Ajuste e Debug:**
    As funções de visualização ajudam na verificação e depuração dos resultados. Sinta-se à vontade para modificar o código conforme necessário para melhor atender às suas necessidades.

## Contato

Em caso de dúvidas ou sugestões, abra uma issue no repositório ou entre em contato. Contribuições para melhorias são sempre bem-vindas!
