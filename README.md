# Parser Monstro — Serviço de Extração de Documentos

Container isolado que extrai texto de documentos de licitação (PDF, DOCX, HTML, imagens, ZIP, RAR, 7z) e expõe uma API REST na porta 7000.

## Requisitos

- Docker Desktop for Windows (ou Docker Engine no Linux/Mac)
- docker-compose v2+

## Como buildar e rodar (Windows)

```bash
# Na pasta parser-service/
docker-compose up --build
```

O serviço ficará disponível em `http://localhost:7000`.

## Testar com curl

### Health check
```bash
curl http://localhost:7000/health
# {"status":"ok","tesseract":true,"easyocr_enabled":false}
```

### Enviar job de parsing (assíncrono)
```bash
curl -X POST http://localhost:7000/parse \
  -H "Content-Type: application/json" \
  -d '{
    "tender_id": "abc-123",
    "documents": [
      {"url": "https://exemplo.gov.br/edital.pdf", "filename": "edital.pdf"}
    ],
    "options": {"enable_easyocr": false}
  }'
# Retorna: {"tender_id":"abc-123","status":"pending","job_id":"<uuid>", ...}
```

### Checar status do job
```bash
curl http://localhost:7000/jobs/<job_id>
```

### Ver fila
```bash
curl http://localhost:7000/queue
```

## Integração com o Monstro (backend principal)

Defina a variável de ambiente `PARSER_URL` no backend:

```env
PARSER_URL=http://localhost:7000
```

No `docker-compose.yml` do projeto principal, adicione ao serviço `backend`:

```yaml
environment:
  - PARSER_URL=http://parser:7000
```

E adicione o serviço `parser` ao mesmo compose (ou use a rede Docker interna).

## Habilitar EasyOCR (opt-in — imagens complexas)

> ⚠️ Aumenta muito o tamanho da imagem (~3GB com PyTorch CPU).

```bash
# Build com EasyOCR
docker-compose build --build-arg ENABLE_EASYOCR=true

# Ou via env no runtime (se já instalado na imagem)
ENABLE_EASYOCR=true docker-compose up
```

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `ENABLE_EASYOCR` | `false` | Habilita EasyOCR como fallback OCR |
| `MAX_WORKERS` | `2` | Jobs simultâneos |
| `LOG_LEVEL` | `INFO` | Nível de log (DEBUG/INFO/WARNING) |
| `STORAGE_ROOT` | `/app/storage` | Pasta de armazenamento temporário |

## Endpoints

| Método | Path | Descrição |
|---|---|---|
| `POST` | `/parse` | Submete job de parsing (retorna `job_id`) |
| `GET` | `/jobs/{job_id}` | Status e resultado do job |
| `GET` | `/health` | Health check |
| `GET` | `/queue` | Contadores da fila |
