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

## EasyOCR

> ⚠️ EasyOCR está temporariamente desabilitado no runtime ROCm deste serviço por instabilidade com meta tensors/device move. Mesmo que `ENABLE_EASYOCR=true` seja enviado, o parser ignora a flag e mantém o fallback OCR em Tesseract.

## Variáveis de ambiente

| Variável | Padrão | Descrição |
|---|---|---|
| `ENABLE_EASYOCR` | `false` | Mantida apenas por compatibilidade; o runtime atual ignora a flag e desabilita EasyOCR |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Default conservador para reduzir fragmentação/OOM em ROCm |
| `HIPBLAS_WORKSPACE_CONFIG` | `:4096:8` | Limita workspace hipBLAS por padrão sem sobrescrever override explícito |
| `RECOGNITION_BATCH_SIZE` | `64` | Default conservador para Surya/Marker em GPU; pode ser sobrescrito no deploy |
| `DETECTOR_BATCH_SIZE` | `8` | Default conservador para Surya/Marker em GPU; pode ser sobrescrito no deploy |
| `MAX_WORKERS` | `2` | Jobs simultâneos |
| `LOG_LEVEL` | `INFO` | Nível de log (DEBUG/INFO/WARNING) |
| `STORAGE_ROOT` | `/app/storage` | Pasta de armazenamento temporário |
| `PARSER_SYSTEMD_UNIT` | vazio | Unidade systemd para consultar com `journalctl -u`; se vazio tenta `INVOCATION_ID` e depois `SYSLOG_IDENTIFIER=parser-monstro` |
| `PARSER_SYSTEM_LOG_PATH` | vazio | Fallback opcional de arquivo de log local quando `journalctl` não estiver disponível/permitido |
| `SYSTEM_LOG_HISTORY_LIMIT` | `500` | Tamanho do buffer em memória usado como último fallback para `/logs/system/recent` |

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

## Nota sobre ROCm + Marker/Surya

O runtime continua **GPU-first**: não há downgrade silencioso para CPU. Para reduzir a chance de OOM depois do docling, o serviço agora:

- aplica defaults de batch mais conservadores para Surya (`RECOGNITION_BATCH_SIZE=64`, `DETECTOR_BATCH_SIZE=8`) quando o deploy não define overrides;
- faz `gc.collect()` + `torch.cuda.empty_cache()` nas fronteiras das etapas GPU;
- registra snapshots de memória/config antes/depois/falha do Marker.

Isso melhora estabilidade, mas não elimina totalmente OOM em PDFs grandes/complexos no stack atual ROCm + marker/surya. Se ainda houver `HIP out of memory`, o próximo passo provável é ajuste adicional de batch/config por deploy ou mudança de versão/configuração upstream da biblioteca.
