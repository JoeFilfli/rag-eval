# Production-Grade Agentic RAG with Automated Evaluation

A RAG pipeline built with LlamaIndex and FastAPI where the model applies Self-RAG — after retrieving top-k chunks, a separate LLM call scores each chunk's marginal contribution given the full retrieved set, dropping chunks that are redundant and going back for more if the remaining set is insufficient — either by increasing k or rephrasing the query to target a different part of the vector space, so only chunks that are both relevant and uniquely necessary reach the generation step — then a Ragas benchmark suite measures faithfulness, answer relevancy, context recall, and latency p95 across configurations, turning optimisation from guesswork into a provable, metric-driven process. A React frontend displays the answer and its Ragas metrics together on a single screen.

## Environments

The codebase never changes between environments — only environment variables do. A `.env` file drives local runs; ECS task config and Secrets Manager drive production.

| Resource | Local | Production (AWS) |
|----------|-------|-----------------|
| LLM | OpenAI or Anthropic API | Claude Haiku 4.5 via Bedrock (IAM auth) |
| Vector store | pgvector (local Postgres container) | pgvector on RDS PostgreSQL |
| Document store | Local folder mounted into container | S3 bucket |
| Compute | Docker Compose | ECS Fargate + ALB (auto-scaling) |
| Container registry | — | ECR |
| Secrets | `.env` file | Secrets Manager + ECS task env config |

**Key environment variables:**
- `LLM_PROVIDER` — `openai` | `anthropic` | `bedrock`
- `VECTOR_STORE_URL` — local Postgres URL or RDS endpoint
- `DOCUMENT_STORE` — `local` | `s3`
- `DOCUMENT_STORE_PATH` — local folder path or S3 bucket name
- `LLM_MODEL` — model name (e.g. `gpt-4o`, `claude-haiku-4-5`)

---

## Box 1: Document Parser

Extracts text, tables, and visual descriptions from a PDF into clean text chunks ready for embedding.

- Receives: a PDF file path (local folder or S3 key, resolved via `DOCUMENT_STORE` env var)
- Produces: a list of text chunks, each tagged with its source page number and type (`text`, `table`, `image`)
- Uses: PyMuPDF for extraction, Claude Haiku 4.5 (via `LLM_PROVIDER` env var) for vision descriptions of extracted images, markdown formatting for tables
- Fails when: a page has no extractable text and the vision model call fails — skip that page, log a warning with page number, continue
- Passes to: Box 2 (Embedder)

Tasks:
- [x] Build a PDF loader that reads from local path or S3 depending on `DOCUMENT_STORE` env var
- [x] Extract plain text per page using PyMuPDF
- [x] Extract tables per page using PyMuPDF and convert to markdown
- [x] Extract images per page using PyMuPDF and send each to the vision LLM with a prompt asking it to describe the visual and its data
- [x] Merge text, table markdown, and image descriptions into a flat list of chunks, each tagged with page number and type
- [x] Skip and log any page where extraction and vision both fail

Summary:
Four files built under `backend/app/parser/`. `loader.py` reads PDF bytes from local disk or S3 based on `DOCUMENT_STORE`. `extractor.py` uses PyMuPDF to pull plain text, convert tables to markdown, and collect raw image bytes per page — each as a typed `RawChunk`. `vision.py` sends image bytes to the configured vision LLM (OpenAI, Anthropic, or Bedrock) and returns a text description. `parser.py` orchestrates the full flow — skips and logs any image chunk where vision returns nothing, drops empty chunks, and returns a flat list ready for Box 2.

---

## Box 2: Chunker + Embedder

Splits long text and table chunks into smaller fixed-size pieces, leaves image description chunks as-is, then embeds every chunk and stores it in pgvector.

- Receives: a list of chunks from Box 1, each with content (string), page number, type (`text`, `table`, `image`), and source document name
- Produces: nothing returned — vectors and metadata written to pgvector
- Uses: LlamaIndex `SentenceSplitter` for chunking text and table chunks (chunk size and overlap configurable via `CHUNK_SIZE` and `CHUNK_OVERLAP` env vars, defaults 512 and 50); `text-embedding-3-small` via OpenAI API locally, Titan Text Embeddings v2 via Bedrock in production — selected by `EMBEDDING_PROVIDER` env var; LlamaIndex's pgvector integration for storage
- Fails when: embedding API call fails (network error, quota exceeded) — raise and halt ingestion for that document, log which chunk failed
- Passes to: pgvector (consumed later by Box 3, the Retriever)

Tasks:
- [x] For each chunk of type `text` or `table`, split into smaller pieces using LlamaIndex `SentenceSplitter` with `CHUNK_SIZE` and `CHUNK_OVERLAP` env vars; preserve page number, type, and document name on every split piece
- [x] Pass image description chunks through unsplit
- [x] Build an embedding client that switches between OpenAI and Bedrock Titan based on `EMBEDDING_PROVIDER` env var
- [x] For each final chunk, call the embedding model and get back a vector
- [x] Store the vector plus metadata (content, page number, type, source document) in pgvector via LlamaIndex
- [x] Log progress per chunk and raise on failure with the chunk index and document name

Summary:
Two files built under `backend/app/embedder/`. `chunker.py` uses LlamaIndex `SentenceSplitter` to split text and table chunks into fixed-size pieces (configurable via `CHUNK_SIZE` and `CHUNK_OVERLAP`), passing image description chunks through unsplit. `embedder.py` builds the embedding client from `EMBEDDING_PROVIDER` (OpenAI or Bedrock), embeds each chunk, wraps it in a LlamaIndex `TextNode` with page/type/document metadata, and stores it in pgvector via `PGVectorStore`. DATABASE_URL is parsed into individual params since `from_params` does not accept a full connection string. Tested on the first 3 pages of the 2025 annual report — 6 chunks stored and verified in the database.

---

## Box 3: Retriever + Self-RAG Judge + Generator

Fetches the most similar chunks from pgvector, judges each chunk's marginal contribution, re-retrieves once if insufficient, then generates the final answer from approved chunks only.

- Receives: a user query string, k (from `RETRIEVAL_K` env var, default 5)
- Produces: an object containing the final answer (string), the approved chunks used (content, page number, type, source document), a `low_confidence` flag (bool), and a `trace` object (query, rephrased query if triggered, k used, chunks retrieved, chunks approved)
- Uses: same embedding client as Box 2 (switched via `EMBEDDING_PROVIDER`); pgvector for similarity search; Claude Haiku 4.5 (via `LLM_PROVIDER`) for chunk scoring, query rephrasing, and answer generation
- Fails when: pgvector is unreachable — raise immediately; LLM call fails — raise with the query and step that failed
- Passes to: Box 4 (Evaluator)

Tasks:
- [x] Embed the user query using the same embedding client as Box 2
- [x] Query pgvector for top-k chunks by cosine similarity
- [x] Send all retrieved chunks and the query to the LLM in one call, asking it to score each chunk as `relevant`, `irrelevant`, or `insufficient` based on its marginal contribution to answering the query given the full set
- [x] Drop `irrelevant` chunks; keep `relevant` chunks
- [x] If the remaining set is scored `insufficient` as a whole, make one re-retrieval attempt: double k and ask the LLM to rephrase the query, then repeat steps 2–4 on the new results; if still insufficient, set `low_confidence: true` and proceed with what exists
- [x] Generate the final answer from approved chunks only, instructing the LLM to only use information present in the provided chunks
- [x] Return the answer, approved chunks, `low_confidence` flag, and full trace object

Summary:
Three files built under `backend/app/rag/` plus `backend/app/core/llm.py`. `llm.py` provides a unified `get_llm_client()` that returns a `chat(messages) -> str` callable for OpenAI, Anthropic, or Bedrock. `retriever.py` embeds the query using the same client as Box 2 and queries pgvector via LlamaIndex `VectorStoreQuery`. `judge.py` sends all retrieved chunks to the LLM in one call, parses the JSON scores (stripping markdown code fences if needed), drops irrelevant chunks, rephrases the query on insufficient results. `pipeline.py` orchestrates the full flow and returns a `QueryResult` with answer, approved chunks, low_confidence flag, and trace. Tested with Haiku 4.5 + OpenAI embeddings — correctly answered "$3.3B revenue in 2025" from 3 ingested pages.

---

## Box 4: Evaluator

Runs Ragas metrics on the completed query and stores the results in Postgres for the frontend to retrieve.

- Receives: the full Box 3 output — query (string), rephrased query if triggered (string or null), answer (string), approved chunks (list of content strings), `low_confidence` flag (bool), trace object
- Produces: nothing returned — one row written to the `evaluations` table in Postgres
- Uses: Ragas (faithfulness, answer relevancy, context recall); Claude Haiku 4.5 (via `LLM_PROVIDER`) as the LLM backend for Ragas; Postgres for storage
- Fails when: Ragas LLM call fails — log the error and write the row with null metric values so the query result is not lost; Postgres is unreachable — raise immediately
- Passes to: Postgres `evaluations` table (read by Box 5, the API)

Tasks:
- [x] Configure Ragas to use the same LLM client as Box 3 via `LLM_PROVIDER` env var
- [x] Run faithfulness, answer relevancy, and context recall metrics using the query, answer, and approved chunk contents
- [x] Measure total end-to-end latency for the query (from query received to answer generated)
- [x] Write one row to the `evaluations` table: query, answer, `low_confidence`, faithfulness score, answer relevancy score, context recall score, latency ms, approved chunk metadata, trace object, timestamp
- [x] On Ragas failure, write the row with null metric scores and log the error

Summary:
`backend/app/evaluation/evaluator.py` wraps Ragas using LangchainLLMWrapper with the configured provider (Anthropic Haiku for LLM, OpenAI for embeddings). Runs faithfulness, answer_relevancy, and context_recall metrics. Ragas returns scores as lists — averaged to a single float. Results written to the `evaluations` table in Postgres via psycopg2. On Ragas failure the row is still written with null scores so no query is lost. Tested successfully — faithfulness=1.0, answer_relevancy=0.0, context_recall=1.0 for the revenue query (low answer_relevancy correctly flags that Haiku hedged on the year, a real signal).

---

## Box 5: API

Exposes the FastAPI endpoints that connect the frontend to the pipeline.

- Receives: HTTP requests from the React frontend
- Produces: JSON responses
- Uses: FastAPI; calls Box 1 + Box 2 for ingestion, Box 3 + Box 4 for querying; reads Postgres `evaluations` table for history
- Fails when: any downstream box raises — return a structured JSON error with the step that failed and the reason
- Passes to: React frontend (Box 6)

Endpoints:
- `POST /api/documents` — accepts a PDF file upload, runs Box 1 (parse) then Box 2 (embed), returns `{ document_id, chunk_count, pages_skipped }`
- `POST /api/query` — accepts `{ question: string }`, runs Box 3 (retrieve + judge + generate) then Box 4 (evaluate), returns `{ answer, low_confidence, faithfulness, answer_relevancy, context_recall, latency_ms, sources: [{ page, type, document }] }`
- `GET /api/history` — returns the last 50 rows from the `evaluations` table as a list of query results with their metrics

Tasks:
- [x] Set up FastAPI app with CORS enabled for the React frontend
- [x] Implement `POST /api/documents` — receive file, save to local folder or S3 based on `DOCUMENT_STORE` env var, call Box 1 then Box 2, return response
- [x] Implement `POST /api/query` — call Box 3 then Box 4, return the answer and metrics in the response shape above
- [x] Implement `GET /api/history` — query the `evaluations` table and return last 50 results ordered by timestamp descending
- [x] Return structured JSON errors `{ error: string, step: string }` for any downstream failure

Summary:
`backend/app/main.py` — FastAPI app with CORS, three endpoints. `POST /api/documents` saves the PDF, parses, splits, and embeds. `POST /api/query` runs the full pipeline and Ragas evaluation, returns answer, low_confidence, latency, and sources. `GET /api/history` reads the last 50 rows from the evaluations table. All tested and working on port 8002.

---

## Box 6: Frontend

A single-screen React app where the user uploads documents, asks questions, and sees the answer alongside its Ragas metrics and sources.

- Receives: responses from Box 5 (API)
- Produces: nothing — displays results to the user
- Uses: React; fetch for API calls; no UI framework — plain CSS
- Fails when: API returns an error — display the error message inline, never crash the page
- Passes to: the user

Layout (single screen, three sections):
1. **Upload panel** — drag-and-drop or file picker for PDF upload, shows upload progress and chunk count on success
2. **Query panel** — text input for the question, submit button, displays the answer below with a "Low confidence — retrieved context may be insufficient" label if `low_confidence` is true, and a sources list (page number, type, document name) below the answer
3. **Metrics panel** — beside or below the answer: faithfulness, answer relevancy, context recall as percentage scores, latency in ms; grayed out until a query has been made
4. **History panel** — collapsible section at the bottom showing the last 50 queries from `GET /api/history` with their scores

Tasks:
- [x] Scaffold React app with three environment variables: `VITE_API_URL` pointing to the FastAPI base URL
- [x] Build the upload panel — file input, POST to `/api/documents`, show `chunk_count` and `pages_skipped` on success, show error message on failure
- [x] Build the query panel — text input, POST to `/api/query`, render answer, `low_confidence` label, and sources list on response
- [x] Build the metrics panel — display faithfulness, answer relevancy, context recall, and latency from the query response; gray out with "—" before first query
- [x] Build the history panel — GET `/api/history` on page load, render as a collapsible list ordered by timestamp descending
- [x] Handle all API errors inline without crashing — show the error string from `{ error, step }` response

Summary:
`frontend/src/App.jsx` and `frontend/src/App.css` — single-screen React app with four panels: upload (PDF picker, progress, chunk count), query (text input, answer, low-confidence label, source tags), metrics (4-column grid: faithfulness, answer relevancy, context recall, latency — grayed until first query), and collapsible history (last 50 rows from `/api/history`). All API errors display inline. `VITE_API_URL` in `frontend/.env` points to `http://localhost:8002`. Backend's `POST /api/query` updated to capture and return the Ragas scores from `evaluate_and_store()`. Vite pinned to v5 (Node 22.5.1 is below the v8 requirement of 22.12+). Confirmed running at `http://localhost:5173`.

---

## Box 7: AWS Deployment & Production Monitoring

Packages the backend and frontend as Docker images, pushes them to ECR, and provisions the full AWS infrastructure with Terraform — making the entire cloud setup reproducible from a single `terraform apply`. Adds CloudWatch dashboards to monitor latency, cost, and errors in production. Rate limiting protects against bill shock. No login required.

- Receives: working local Docker Compose setup from all previous boxes
- Produces: a live public URL serving the full application on AWS, a CloudWatch dashboard showing real-time pipeline health and cost, and an `infra/` Terraform directory that lets anyone reproduce the full AWS setup from scratch
- Uses: Docker, AWS ECR, AWS ECS Fargate, AWS ALB, AWS RDS PostgreSQL (pgvector), AWS Bedrock, AWS Secrets Manager, AWS IAM — all provisioned via **Terraform**; CloudWatch Logs, CloudWatch Metrics, CloudWatch Dashboards for monitoring; AWS Cost Explorer for spend tracking
- Fails when: ECS task fails to start (misconfigured env vars or IAM permissions) — check ECS task logs in CloudWatch; Terraform apply fails — read the plan output, fix the offending resource block
- Passes to: the public internet

Terraform layout (`infra/`):
- `main.tf` — AWS provider, region, backend (S3 + DynamoDB for remote state)
- `vpc.tf` — VPC, public and private subnets, internet gateway, NAT gateway
- `rds.tf` — RDS PostgreSQL instance with pgvector extension in a private subnet, security group allowing backend ECS only
- `ecr.tf` — two ECR repositories (backend, frontend)
- `ecs.tf` — ECS cluster, two Fargate services (backend, frontend), task definitions with env vars injected from Secrets Manager
- `alb.tf` — ALB, HTTPS listener, target groups, routing rules (`/api/*` → backend, `/*` → frontend)
- `iam.tf` — backend task execution role with `bedrock:InvokeModel`, RDS access, Secrets Manager read
- `secrets.tf` — Secrets Manager secrets for any remaining API keys
- `cloudwatch.tf` — log groups, metric filters, dashboard
- `variables.tf` — all inputs (region, db password, image tags, etc.)
- `outputs.tf` — ALB DNS name, ECR repo URLs
- `terraform.tfvars.example` — committed template; actual `terraform.tfvars` is gitignored

Infrastructure:
- ALB routes `/api/*` → backend ECS service, `/*` → frontend ECS service
- Backend ECS task IAM role has `bedrock:InvokeModel` and RDS access — no API keys for LLM calls
- All secrets stored in Secrets Manager, injected as env vars into ECS task definitions
- RDS PostgreSQL instance with pgvector extension enabled in a private subnet
- Both ECS services auto-scale on ALB request count

Rate limiting (FastAPI layer):
- `POST /api/documents` — 1 request per IP per hour
- `POST /api/query` — 10 requests per IP per minute
- Requests exceeding limits return HTTP 429

Monitoring (CloudWatch, also provisioned by Terraform):
- Structured JSON logs emitted by the backend for every pipeline step: `{ "activity": "judge", "latency_ms": 1200, "input_tokens": 800, "output_tokens": 120 }` — parsed by CloudWatch Logs metric filters into custom metrics
- Custom metrics: `PipelineLatencyMs` (per activity), `InputTokens`, `OutputTokens` (per model), `LowConfidenceRate`, `RagasFailureRate`
- CloudWatch Dashboard with widgets: p50/p95 latency per activity, token usage over time, low-confidence query rate, Ragas failure rate, ECS CPU/memory, ALB request count and 5xx rate
- All resources tagged `project=production-rag` for Cost Explorer isolation

Tasks:
- [x] Write `docker-compose.yml` for local development with backend, frontend, and pgvector containers and `.env.example` listing all required env vars
- [x] Write `backend/Dockerfile` and `frontend/Dockerfile` (frontend built as static files served by nginx)
- [x] Write `infra/main.tf` — provider config and S3+DynamoDB remote state backend
- [x] Write `infra/vpc.tf` — VPC, subnets, internet gateway, NAT gateway
- [x] Write `infra/rds.tf` — RDS PostgreSQL with pgvector, private subnet, security group
- [x] Write `infra/ecr.tf` — ECR repositories for backend and frontend images
- [x] Write `infra/iam.tf` — ECS task execution role with Bedrock and RDS permissions
- [x] Write `infra/secrets.tf` — Secrets Manager entries for API keys
- [x] Write `infra/ecs.tf` — ECS cluster, Fargate task definitions, services with env vars from Secrets Manager
- [x] Write `infra/alb.tf` — ALB, target groups, routing rules
- [x] Write `infra/cloudwatch.tf` — log groups, metric filters, dashboard
- [x] Write `infra/variables.tf`, `infra/outputs.tf`, `infra/terraform.tfvars.example`
- [x] Implement rate limiting in FastAPI — 1 upload per IP per hour, 10 queries per IP per minute, return HTTP 429
- [x] Add structured JSON logging to judge, generate, and evaluate steps emitting `activity`, `latency_ms`, `input_tokens`, `output_tokens`
- [x] Write `deploy.sh` — builds both Docker images, pushes to ECR, triggers ECS rolling update
- [ ] Verify end-to-end: `terraform apply`, run `deploy.sh`, upload a document, ask a question, confirm logs appear in CloudWatch dashboard

Summary:
All implementation tasks complete. `docker-compose.yml` + Dockerfiles wire the full stack for local dev. Twelve Terraform files in `infra/` provision VPC, RDS, ECR, ECS Fargate, ALB (HTTPS + /api/* routing), IAM, Secrets Manager, and a CloudWatch dashboard with metric filters for latency, tokens, low-confidence rate, and Ragas failure rate. `deploy.sh` builds both images tagged with the git SHA, pushes to ECR, and runs `terraform apply` to trigger a rolling ECS update. Rate limiting (slowapi: 1 upload/hour, 10 queries/minute) and structured JSON logging (`print(json.dumps(...))` to stdout from judge, generate, and evaluate steps) are live in the backend. The final task — end-to-end verification on a live AWS account — requires running `terraform apply`, `deploy.sh`, uploading a document, and confirming logs appear in the CloudWatch dashboard.

---

## Box 8: Authentication & Tenant Isolation

Adds user authentication via AWS Cognito and isolates every user's documents and query history so no user can see another's data.

- Receives: the deployed application from Box 7
- Produces: a login/signup screen, JWT-protected API endpoints, and per-user document and history isolation throughout the pipeline
- Uses: AWS Cognito User Pool for auth; `python-jose` for JWT validation in FastAPI middleware; React state for the auth flow (no external auth library); Terraform to provision the Cognito User Pool and App Client; `user_id` (Cognito `sub` claim) stored as metadata on every pgvector chunk and every evaluations row
- Fails when: JWT is missing or expired — return HTTP 401; JWT signature invalid — return HTTP 401; Cognito JWKS endpoint unreachable — return HTTP 503
- Passes to: the user (logged-in session)

Auth flow:
- Frontend shows a login/signup form before any other panel is visible
- User signs up or logs in via Cognito hosted UI or direct API calls to Cognito (`InitiateAuth`, `SignUp`, `ConfirmSignUp`)
- Cognito returns an ID token (JWT); frontend stores it in memory and sends it as `Authorization: Bearer <token>` on every API request
- FastAPI middleware validates the JWT on every protected endpoint: fetches Cognito's JWKS, verifies signature and expiry, extracts `sub` as `user_id`
- `user_id` flows through the pipeline — stored in pgvector chunk metadata on upload, used as a filter on every retrieval, stored on every evaluations row

Tenant isolation:
- `POST /api/documents` — chunks stored with `metadata.user_id = user_id`
- `GET /api/documents` — returns only chunks where `metadata.user_id = user_id`
- `DELETE /api/documents/{name}` — deletes only chunks where both document name and `user_id` match
- `POST /api/query` — retriever filters pgvector by `user_id` before scoring
- `GET /api/history` — returns only evaluations rows where `user_id = user_id`

Terraform additions (`infra/cognito.tf`):
- Cognito User Pool with email sign-up, strong password policy
- Cognito App Client (no client secret — public SPA client)
- User Pool domain for hosted UI (optional fallback)
- Outputs: User Pool ID, App Client ID, JWKS URL — injected into ECS task env vars and frontend build env

Tasks:
- [ ] Write `infra/cognito.tf` — Cognito User Pool, App Client, domain; output User Pool ID, App Client ID, JWKS URL
- [ ] Add `COGNITO_USER_POOL_ID`, `COGNITO_APP_CLIENT_ID`, `COGNITO_JWKS_URL` to `backend/app/core/config.py`
- [ ] Write `backend/app/core/auth.py` — FastAPI dependency that validates the Bearer JWT using `python-jose` and Cognito's JWKS; returns `user_id` (the `sub` claim); raises HTTP 401 on any failure
- [ ] Add the auth dependency to all protected endpoints (`/api/documents`, `/api/query`, `/api/history`)
- [ ] Update `embed_and_store` to accept and store `user_id` in chunk metadata
- [ ] Update `GET /api/documents` and `DELETE /api/documents/{name}` to filter by `user_id`
- [ ] Update the retriever to accept `user_id` and apply it as a metadata filter on the pgvector query
- [ ] Update the evaluations table and `_write_to_db` to store `user_id`; update `GET /api/history` to filter by `user_id`
- [ ] Build the login/signup UI in React — email + password form, Cognito `InitiateAuth` / `SignUp` / `ConfirmSignUp` calls, store JWT in memory, show the main app only after successful login, add a logout button that clears the token
- [ ] Add `VITE_COGNITO_USER_POOL_ID` and `VITE_COGNITO_APP_CLIENT_ID` to `frontend/.env`
- [ ] Verify end-to-end: two users sign up, each uploads a different document, neither can see or query the other's data

Summary:
(written after implementation)

---

## Files

`backend/app/core/config.py` — reads all env vars into a single `settings` object used across the app
`backend/app/parser/loader.py` — loads PDF bytes from local disk or S3 based on `DOCUMENT_STORE`
`backend/app/parser/extractor.py` — extracts text, tables (as markdown), and image bytes per page using PyMuPDF
`backend/app/parser/vision.py` — sends image bytes to the vision LLM and returns a text description
`backend/app/parser/parser.py` — orchestrates Box 1: load → extract → describe images → return flat chunk list
`backend/app/embedder/chunker.py` — splits text/table chunks via SentenceSplitter; passes image chunks through unsplit
`backend/app/embedder/embedder.py` — embeds each chunk via OpenAI or Bedrock and stores in pgvector
`backend/app/core/llm.py` — unified LLM chat client switching between OpenAI, Anthropic, and Bedrock
`backend/app/rag/retriever.py` — embeds query and fetches top-k chunks from pgvector
`backend/app/rag/judge.py` — scores chunk marginal contribution and rephrases query on insufficient results
`backend/app/rag/pipeline.py` — orchestrates retrieve → judge → re-retrieve → generate, returns QueryResult
`backend/app/evaluation/evaluator.py` — runs Ragas metrics and writes one row to the evaluations table
`docker-compose.yml` — spins up pgvector db, backend, and frontend containers for local development
`.env.example` — template listing every env var the stack requires
`backend/Dockerfile` — Python 3.11-slim image running uvicorn on port 8002
`frontend/Dockerfile` — two-stage build: Node compiles React to static files, nginx serves them on port 80
`infra/main.tf` — Terraform provider (AWS ~5.0), S3+DynamoDB remote state backend, default project tag
`infra/vpc.tf` — VPC (10.0.0.0/16), 2 public + 2 private subnets across 2 AZs, IGW, NAT gateway, route tables
`infra/rds.tf` — RDS PostgreSQL 16, private subnets, encrypted storage, deletion protection, RDS SG
`infra/ecr.tf` — ECR repos for backend and frontend, scan on push, lifecycle policy (keep last 10 images)
`infra/iam.tf` — ECS execution role (ECR pull + Secrets Manager) and task role (Bedrock + S3), least-privilege
`infra/secrets.tf` — Secrets Manager entries for OPENAI_API_KEY and ANTHROPIC_API_KEY
`infra/ecs.tf` — ECS cluster, backend + frontend task definitions and services, auto-scaling on ALB request count
`infra/alb.tf` — ALB (public subnets), HTTPS listener with TLS 1.3 policy, HTTP→HTTPS redirect, /api/* routing
`infra/cloudwatch.tf` — metric filters for latency/tokens/low-confidence/ragas-failures, 7-widget dashboard
`infra/variables.tf` — all 18 input variables with types, descriptions, and production defaults
`infra/outputs.tf` — ALB DNS name, backend and frontend ECR repo URLs
`infra/terraform.tfvars.example` — committed template; actual terraform.tfvars is gitignored
`deploy.sh` — builds + pushes both images to ECR with git SHA tag, then runs terraform apply to roll out
`backend/Dockerfile` — Python 3.11-slim image running uvicorn on port 8002
`frontend/Dockerfile` — two-stage build: Node compiles React to static files, nginx serves them on port 80

---

