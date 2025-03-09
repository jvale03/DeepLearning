import os

# Defina o caminho da pasta onde estão os arquivos PDF
folder_path = "scraper/allPapers"

# Nome do arquivo de saída
output_file = "lista_pdfs.txt"

# Obtém a lista de arquivos PDF
pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]

# Escreve no arquivo de texto
with open(output_file, "w") as f:
    for pdf in pdf_files:
        f.write(pdf + "\n")

print(f"Arquivo '{output_file}' criado com sucesso, contendo {len(pdf_files)} arquivos PDF.")
