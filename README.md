# Lobeira — Serviço de Extração de Documentos

Serviço isolado que extrai texto de documentos de licitação (PDF, DOCX, HTML, imagens, ZIP, RAR, 7z) e expõe uma API REST na porta 7001.

## Requisitos

- Python 3.11+
- Tesseract OCR instalado (`tesseract-ocr tesseract-ocr-por`)
- GPU AMD com ROCm 7.2+ para o PaddleOCR-VL-1.5 (opcional, tem fallback CPU)
- vLLM com backend ROCm para o servidor PaddleOCR-VL-1.5

## Como rodar

```bash
cd ~/Lobeira
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 7001
```

O serviço ficará disponível em `http://localhost:7001`.

## Testar com curl

### Health check
```bash
curl http://localhost:7001/health
# {"status":"ok","tesseract":true,...}
```

### Enviar job de parsing (assíncrono)
```bash
curl -X POST http://localhost:7001/parse \
  -H "Content-Type: application/json" \
  -d '{
    "tender_id": "abc-123",
    "documents": [
      {"url": "https://exemplo.gov.br/edital.pdf", "filename": "edital.pdf"}
    ]
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

### Ver logs recentes do job store
```bash
curl http://localhost:7000/logs/recent
curl http://localhost:7000/logs/job/<job_id>
```

### Ver logs do sistema (journald/systemd com fallback)
```bash
curl "http://localhost:7000/logs/system/recent?limit=50&contains=rocm"
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

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `PADDLE_OCR_URL` | `http://127.0.0.1:8100` | Endpoint do servidor vLLM com PaddleOCR-VL-1.5 |
| `PADDLE_OCR_MODEL` | `PaddlePaddle/PaddleOCR-VL-1.5` | Nome do modelo servido pelo vLLM |
| `PADDLE_OCR_TIMEOUT` | `120` | Timeout em segundos por página |
| `PADDLE_OCR_GPU_UTIL` | `0.85` | GPU memory utilization do vLLM |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | Reduz fragmentação/OOM em ROCm |
| `HSA_OVERRIDE_GFX_VERSION` | `11.0.0` | Necessário para AMD RX 7800 XT (gfx1101/RDNA3) |
| `MAX_WORKERS` | `1` | Jobs simultâneos |
| `LOG_LEVEL` | `INFO` | Nível de log (DEBUG/INFO/WARNING) |
| `STORAGE_ROOT` | `/app/storage` | Pasta de armazenamento; páginas problemáticas vão em `review/{job_id}/` |
| `PARSER_SYSTEMD_UNIT` | vazio | Unidade systemd para `journalctl -u` |
| `PARSER_SYSTEM_LOG_PATH` | vazio | Fallback de arquivo de log local |
| `SYSTEM_LOG_HISTORY_LIMIT` | `500` | Buffer em memória para `/logs/system/recent` |

## Endpoints

| Método | Path | Descrição |
|---|---|---|
| `POST` | `/parse` | Submete job de parsing (retorna `job_id`) |
| `GET` | `/jobs/{job_id}` | Status e resultado do job |
| `GET` | `/health` | Health check |
| `GET` | `/queue` | Contadores da fila |
| `GET` | `/logs/recent` | Jobs recentes com logs resumidos |
| `GET` | `/logs/job/{job_id}` | Logs detalhados de um job, incluindo documentos |
| `GET` | `/logs/system/recent` | Logs estruturados do runtime via `journalctl`, com filtros `limit`, `since`, `contains` e exclusão padrão de access logs |


## Observabilidade de logs do sistema

O endpoint `GET /logs/system/recent` foi pensado para o cenário real em Linux com systemd/journald. Ele tenta, nesta ordem:

1. `journalctl` em JSON (`INVOCATION_ID` atual quando disponível, senão `PARSER_SYSTEMD_UNIT`, senão `SYSLOG_IDENTIFIER=parser-monstro`)
2. arquivo local definido por `PARSER_SYSTEM_LOG_PATH`
3. buffer em memória do processo atual

Por padrão o endpoint remove linhas ruidosas de access log/health/queue/logs para destacar mensagens úteis de runtime (ex.: ROCm, fallback, init de modelos, OOM, warnings). Use `include_access_logs=true` para ver tudo. O parser também normaliza mensagens não-string vindas do journal (listas, bytes, dicts em JSON), evitando o fallback espúrio com warning `expected string or bytes-like object`.

### Caveat de permissão

Em alguns deploys o processo do parser pode não ter permissão para ler o journal do host. Nesse caso o endpoint não quebra: ele retorna `warnings` explicando a falha e usa o fallback disponível. Para ter a visão mais completa, garanta que o serviço possa executar `journalctl` com acesso ao journal correspondente, ou configure `PARSER_SYSTEM_LOG_PATH`.

## Pipeline de extração

```
1. PyMuPDF          → PDF nativo com texto?       → ✅ suficiente (score ≥ 0.9) → para
   ↓ não
2. PaddleOCR-VL-1.5 → PDF escaneado/complexo (GPU)→ ✅ suficiente (score ≥ 0.82) → para
   ↓ timeout / OOM / confiança baixa
3. Tesseract `por`  → CPU, sem GPU               → ✅ suficiente (score ≥ 0.5) → para
   ↓ resultado vazio ou abaixo do threshold
4. Falha registrada + páginas problemáticas salvas em STORAGE_ROOT/review/{job_id}/
```

O PaddleOCR-VL-1.5 roda como servidor vLLM separado (ver `vl-ocr.service`). Se o servidor não estiver disponível, o pipeline cai automaticamente para Tesseract.

## Setup do PaddleOCR-VL-1.5 (servidor vLLM)

```bash
# Variáveis de ambiente permanentes (~/.bashrc ou /etc/environment)
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Instalar vLLM com ROCm
pip install vllm --extra-index-url https://download.pytorch.org/whl/rocm6.2

# Instalar e iniciar o serviço systemd
sudo cp vl-ocr.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable vl-ocr
sudo systemctl start vl-ocr
```

O serviço vLLM ficará em `http://127.0.0.1:8100`.
