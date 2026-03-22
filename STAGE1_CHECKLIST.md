# STAGE 1 EXTRACTION — WAKE-UP CHECKLIST

## Pre-Flight (5 minutes)
- [ ] `HF_TOKEN` environment variable set (HuggingFace write token)
- [ ] `HF_REPO_ID` environment variable set (e.g. `Jibbalit/stage1_reasoning_corpus`)
- [ ] RunPod account funded ($200–300) **OR** local machine with 16+ vCPU / 32 GB RAM
- [ ] Repository cloned locally

## Deploy (10 minutes)
- [ ] Verify `stage1_manifest.csv` is present (73 documents, all `tier=1`)
- [ ] Create HuggingFace dataset repo: `Jibbalit/stage1_reasoning_corpus`
- [ ] Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- [ ] Test one document in dry-run mode:
  ```bash
  DRY_RUN=true python main.py --stage1
  ```
- [ ] Confirm no import errors and at least one record printed/logged

## Scale (if local test passes)
- [ ] Spin up RunPod instance (CPU, 16 vCPU, 32 GB RAM)
- [ ] Upload repository to instance
- [ ] Run full Stage 1 extraction:
  ```bash
  HF_TOKEN=<token> HF_REPO_ID=Jibbalit/stage1_reasoning_corpus python main.py --stage1
  ```
- [ ] Monitor HuggingFace dataset viewer for growing file count

## Validate (ongoing)
- [ ] Check uploaded JSONL files in HF dataset viewer
- [ ] Sample 3–5 documents for content quality
- [ ] Confirm `tier=1` and `structural_prior` tags present in metadata
- [ ] Run validation tests:
  ```bash
  python -m pytest tests/test_pipeline.py -v
  ```
- [ ] Total documents extracted: **target 73**
- [ ] Total tokens: **target 500 K – 1 M**

## Extractor Status

| extractor_type | Class | Status |
|---|---|---|
| DarwinXML | DarwinExtractor | ✅ implemented |
| NewtonXML | DarwinExtractor | ✅ implemented |
| PlatoText | PlatoExtractor | ✅ implemented |
| GalileoText | PlatoExtractor | ✅ implemented |
| AristotleText | PlatoExtractor | ✅ implemented |
| HerschelText | PlatoExtractor | ✅ implemented |
| FeynmanHTML | DialogueExtractor | ✅ implemented |
| EuclidHTML | DialogueExtractor | ✅ implemented |
| HTMLText | DialogueExtractor | ✅ implemented |
| MahajanHTML | DialogueExtractor | ✅ implemented |
| PDFText | PDFAcademicExtractor | ✅ implemented |
| ExtractPDF | PDFAcademicExtractor | ✅ implemented |
| MahajanPDF | PDFAcademicExtractor | ✅ implemented |
| CyberneticsOpenAccess | PDFAcademicExtractor | ✅ implemented |
| AutodeskDocs | DialogueExtractor | ✅ implemented |
| OpenSCADDocs | DialogueExtractor | ✅ implemented |
| BlenderDocs | DialogueExtractor | ✅ implemented |
| GitHubJupyter | (requires repo slugs) | ⚠️ skipped |
| GitHubMarkdown | (requires repo slugs) | ⚠️ skipped |
| ExtractBook | (copyrighted — manual) | ⚠️ skipped |

## Done when:
- [ ] 73 Stage 1 documents extracted and uploaded
- [ ] All 28 pipeline tests pass (`python -m pytest tests/`)
- [ ] Total tokens ~500 K–1 M
- [ ] Ready for Stage 2
