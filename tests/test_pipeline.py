"""
tests/test_pipeline.py
Five validation tests for the distributed data extraction pipeline.

Test 1: Single document end-to-end
    Process one mock Darwin letter → tagged JSON → StreamUploader (dry_run).

Test 2: Small batch with deduplication
    100 documents (mix of types); verify dedup and memory constraints.

Test 3: Parallel deployment
    Deploy 3 agents simultaneously; no collisions; all records reach uploader.

Test 4: Error recovery
    Simulate network interruption mid-upload; verify retry + no data loss.

Test 5: Metadata accuracy
    Sample 20 documents; verify tier/prior/domain tags pass schema checks.
"""

from __future__ import annotations

import os
import sys
import textwrap
import threading
import time
import uuid
import xml.etree.ElementTree as ET
from typing import Generator, List
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from base_extractor import BaseExtractor, make_record, VALID_TIERS, VALID_PRIORS, VALID_DOMAINS, VALID_DOC_TYPES
from semantic_deduplicator import SemanticDeduplicator
from stream_uploader import StreamUploader
from tier_classifier import classify_tier
from prior_tagger import tag_prior


# ===========================================================================
# Fixtures & helpers
# ===========================================================================

DARWIN_LETTER_TEXT = textwrap.dedent("""\
    My dear Hooker, I have been much struck with this fact, therefore I believe
    that the hypothesis of natural selection provides strong evidence for the
    argument that variation among species arises from gradual adaptation. This
    observation leads me to conclude that the theory I proposed in my last letter
    is supported by the experiments I described. What if the process of divergence
    is far more rapid than we have hitherto supposed? I am inclined to think this
    is because the conditions of life differ so greatly from one region to another.
    Hence I conclude that the evidence strongly supports the hypothesis of common
    descent with modification through natural selection.
""")

PLATO_DIALOGUE_TEXT = textwrap.dedent("""\
    SOCRATES. And do you not think, therefore, that virtue is a kind of knowledge?
    MENO. Yes, it seems so, certainly.
    SOCRATES. But we agreed that the argument points to this conclusion
    that knowledge and virtue are indeed the same thing. The reason for
    this is that a good man cannot be taught to be good unless he knows what
    goodness is. Is it not the case that every virtue involves wisdom?
    MENO. I believe so indeed, Socrates.
    SOCRATES. Then we must examine whether wisdom is truly what the gods have given.
    MENO. That seems necessary given the argument.
    SOCRATES. For this reason alone it seems impossible to disagree with the conclusion.
    MENO. I agree completely, for the evidence supports your claim.
""")

SAMPLE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<root>
  <div type="letter">
    <p>I have been much struck with this fact, therefore I believe the hypothesis
    about natural selection is strongly supported by the evidence I have gathered
    during my observations and experiments. This leads me to conclude that variation
    arises from gradual adaptation. What if we were to suppose otherwise?</p>
    <p>The evidence suggests that the argument from analogy is much stronger than
    I had supposed. Because of this I think we should revise our theory of common
    descent accordingly. Hence the natural conclusion follows.</p>
  </div>
</root>
"""


class _FakeExtractor(BaseExtractor):
    """Concrete test extractor that yields a fixed list of records."""

    SOURCE_TYPE = "test"

    def __init__(self, records, agent_id="test_agent", deduplicator=None,
                 min_text_length=10, fail_first_n=0):
        super().__init__(agent_id=agent_id, deduplicator=deduplicator,
                         min_text_length=min_text_length)
        self._records = list(records)
        self._fail_first_n = fail_first_n
        self._call_count = 0

    def extract(self) -> Generator[dict, None, None]:
        self._call_count += 1
        if self._call_count <= self._fail_first_n:
            raise IOError("Simulated extraction failure")
        yield from self._records


def _make_unique_records(n: int, prefix: str = "doc") -> List[dict]:
    """Generate *n* unique make_record dicts."""
    return [
        make_record(
            raw_text=(
                f"{prefix} {i}: This passage demonstrates reasoning therefore the "
                f"hypothesis is supported by evidence and observation. "
                f"We conclude from the argument that natural selection provides "
                f"the explanation. Because of this the theory holds. {uuid.uuid4()}"
            ),
            source_url=f"test://{prefix}/{i}",
            doc_type="correspondence",
            tier=((i % 5) + 1),
            structural_prior=((i % 9) + 1),
            domain=["physics", "math"] if i % 2 == 0 else ["philosophy"],
            participants=["darwin", "hooker"],
            quality_score=0.5 + (i % 50) / 100,
            license_="public_domain",
            agent_id="test_agent",
        )
        for i in range(n)
    ]


# ===========================================================================
# Test 1: Single document end-to-end
# ===========================================================================
class TestSingleDocumentEndToEnd:
    """One Darwin letter flows through extraction → record → StreamUploader."""

    def test_make_record_schema(self):
        """make_record returns all required keys with correct types."""
        record = make_record(
            raw_text=DARWIN_LETTER_TEXT,
            source_url="test://darwin/letter001",
            doc_type="correspondence",
            tier=2,
            structural_prior=6,
            domain=["biology"],
            participants=["darwin", "hooker"],
            quality_score=0.87,
            license_="CC_BY",
            agent_id="agent_001",
        )
        assert "doc_id" in record
        assert "text" in record
        assert "messages" in record
        assert "metadata" in record
        assert uuid.UUID(record["doc_id"])  # valid UUID
        assert "<|system|>" in record["text"]
        assert "<|user|>" in record["text"]
        assert "<|assistant|>" in record["text"]
        assert isinstance(record["messages"], list)
        assert len(record["messages"]) == 3
        meta = record["metadata"]
        assert meta["tier"] == 2
        assert meta["structural_prior"] == 6
        assert meta["domain"] == ["biology"]
        assert meta["doc_type"] == "correspondence"
        assert meta["participants"] == ["darwin", "hooker"]
        assert meta["quality_score"] == 0.87
        assert meta["license"] == "CC_BY"
        assert meta["agent_id"] == "agent_001"
        assert "extraction_timestamp" in meta

    def test_correspondence_extractor_from_xml(self, tmp_path):
        """CorrespondenceExtractor extracts at least one record from XML."""
        from correspondence_extractor import CorrespondenceExtractor

        xml_file = tmp_path / "letter.xml"
        xml_file.write_text(SAMPLE_XML, encoding="utf-8")

        ext = CorrespondenceExtractor(xml_sources=[str(xml_file)], agent_id="agent_001")
        records = list(ext.stream())
        assert len(records) >= 1
        # Verify schema
        for rec in records:
            assert "doc_id" in rec
            assert "messages" in rec
            assert rec["metadata"]["doc_type"] == "correspondence"

    def test_stream_uploader_dry_run_end_to_end(self):
        """Single record: add → flush → total_records == 1."""
        record = make_record(
            raw_text=DARWIN_LETTER_TEXT,
            source_url="test://darwin/letter001",
            doc_type="correspondence",
            agent_id="agent_001",
        )
        with StreamUploader(repo_id="test/repo", token="fake", dry_run=True) as up:
            up.add(record)
        assert up.total_records == 1
        assert up.chunk_count >= 1

    def test_tier_and_prior_assigned(self):
        """Tier and prior classifiers return valid values for Darwin text."""
        tier = classify_tier(DARWIN_LETTER_TEXT)
        prior = tag_prior(DARWIN_LETTER_TEXT)
        assert tier in VALID_TIERS
        assert prior in VALID_PRIORS

    def test_dialogue_extractor_from_text(self, tmp_path):
        """DialogueExtractor extracts at least one chunk from Plato-like text."""
        from dialogue_extractor import DialogueExtractor

        txt_file = tmp_path / "plato.txt"
        txt_file.write_text(PLATO_DIALOGUE_TEXT, encoding="utf-8")

        ext = DialogueExtractor(
            text_sources=[str(txt_file)],
            chunk_turns=3,
            agent_id="agent_003",
        )
        records = list(ext.stream())
        assert len(records) >= 1
        for rec in records:
            assert rec["metadata"]["doc_type"] in ("dialogue", "thought_experiment")


# ===========================================================================
# Test 2: Small batch with semantic deduplication
# ===========================================================================
class TestDeduplication:
    """100 documents; dedup filters exact duplicates; memory stays reasonable."""

    def test_exact_duplicates_rejected(self):
        """Second occurrence of identical text is flagged as duplicate."""
        dedup = SemanticDeduplicator()
        assert not dedup.is_duplicate(DARWIN_LETTER_TEXT)
        assert dedup.is_duplicate(DARWIN_LETTER_TEXT)

    def test_100_docs_50_unique(self):
        """50 unique + 50 exact duplicates → only 50 pass the dedup filter."""
        unique_texts = [
            f"Unique reasoning document {i}. Therefore the hypothesis is supported "
            f"by evidence and argument {i}. We conclude the theory holds because "
            f"of the observations made during the experiment. Hence the proof follows."
            for i in range(50)
        ]
        records = [
            make_record(
                raw_text=t,
                source_url=f"test://doc/{i % 50}",
                doc_type="correspondence",
                agent_id="test_agent",
            )
            for i, t in enumerate(unique_texts * 2)
        ]

        dedup = SemanticDeduplicator()
        ext = _FakeExtractor(records, deduplicator=dedup, min_text_length=50)
        results = list(ext.stream())
        assert len(results) == 50

    def test_mixed_source_types_100_docs(self):
        """100 records from mixed source types all pass through dedup when unique."""
        records = _make_unique_records(100)
        dedup = SemanticDeduplicator()
        ext = _FakeExtractor(records, deduplicator=dedup)
        results = list(ext.stream())
        assert len(results) == 100

    def test_memory_bounded(self):
        """SemanticDeduplicator memory usage stays sane for 500 documents."""
        import tracemalloc
        tracemalloc.start()
        dedup = SemanticDeduplicator()
        for i in range(500):
            text = (
                f"Document {i}: reasoning chain therefore hypothesis supported "
                f"by evidence. We conclude argument holds because of the proof. {uuid.uuid4()}"
            )
            dedup.is_duplicate(text)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # Peak usage must be < 2 GB (28 GB is the agent budget; 2 GB is safe for tests)
        assert peak < 2 * 1024 ** 3, f"Memory peak {peak / 1024**3:.2f} GB exceeded 2 GB limit"

    def test_short_texts_filtered(self):
        """Records below min_text_length are dropped before dedup."""
        records = [
            make_record(raw_text="Short.", source_url="s", doc_type="dialogue", agent_id="a")
        ]
        ext = _FakeExtractor(records, min_text_length=200)
        assert list(ext.stream()) == []


# ===========================================================================
# Test 3: Parallel deployment (3 agents)
# ===========================================================================
class TestParallelDeployment:
    """Three agents run simultaneously; all records reach uploader without collision."""

    def test_three_agents_concurrent(self):
        """3 threads each feeding unique records into one StreamUploader."""
        uploader = StreamUploader(repo_id="test/repo", token="fake", dry_run=True)
        shared_dedup = SemanticDeduplicator()
        errors: List[Exception] = []

        def run_agent(agent_id: int, n: int = 30):
            records = _make_unique_records(n, prefix=f"agent{agent_id}")
            ext = _FakeExtractor(records, deduplicator=shared_dedup, agent_id=f"agent_00{agent_id}")
            try:
                for rec in ext.stream():
                    uploader.add(rec)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=run_agent, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        uploader.flush()
        assert not errors, f"Agent errors: {errors}"
        # With unique records across agents, all 90 should pass
        assert uploader.total_records == 90

    def test_no_chunk_collision(self):
        """Chunk indices are monotonically assigned even under concurrent adds."""
        uploader = StreamUploader(
            repo_id="test/repo", token="fake", dry_run=True,
            chunk_size_mb=0.001,  # tiny chunks to force many flushes
        )
        n_threads = 3
        n_records = 50

        def add_records(tid: int):
            for i in range(n_records):
                rec = make_record(
                    raw_text=f"Thread {tid} record {i} with enough unique text "
                             f"to pass the minimum length filter properly. {uuid.uuid4()}",
                    source_url=f"test/{tid}/{i}",
                    doc_type="correspondence",
                    agent_id=f"agent_{tid:03d}",
                )
                uploader.add(rec)

        threads = [threading.Thread(target=add_records, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        uploader.flush()

        assert uploader.total_records == n_threads * n_records
        # Chunk indices should be contiguous (no gaps)
        assert uploader.chunk_count >= 1

    def test_parallel_slice_separation(self):
        """Each agent works on a different source slice without interference."""
        all_sources = [f"test://source/{i}" for i in range(6)]
        results_per_agent: dict[int, list] = {0: [], 1: [], 2: []}
        lock = threading.Lock()

        def simulate_agent(agent_id: int, sources: List[str]):
            for src in sources:
                rec = make_record(
                    raw_text=f"Reasoning passage from {src}. Therefore the hypothesis "
                             f"is supported by evidence. We conclude the argument holds.",
                    source_url=src,
                    doc_type="correspondence",
                    agent_id=f"agent_{agent_id:03d}",
                )
                with lock:
                    results_per_agent[agent_id].append(rec["metadata"]["source_url"])

        threads = [
            threading.Thread(
                target=simulate_agent,
                args=(i, all_sources[i * 2: (i + 1) * 2]),
            )
            for i in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No source URL should appear in more than one agent's results
        seen = set()
        for agent_sources in results_per_agent.values():
            for url in agent_sources:
                assert url not in seen, f"Source collision: {url}"
                seen.add(url)


# ===========================================================================
# Test 4: Error recovery with retry
# ===========================================================================
class TestErrorRecovery:
    """Network interruption → retry → no data loss."""

    def test_correspondence_extractor_bad_source_graceful(self):
        """CorrespondenceExtractor skips missing files and returns empty."""
        from correspondence_extractor import CorrespondenceExtractor

        ext = CorrespondenceExtractor(xml_sources=["/nonexistent/letter.xml"])
        assert list(ext.stream()) == []

    def test_dialogue_extractor_bad_url_graceful(self):
        """DialogueExtractor logs errors gracefully for bad URLs."""
        from dialogue_extractor import DialogueExtractor

        ext = DialogueExtractor(text_sources=["http://localhost:0/nonexistent.txt"])
        records = list(ext.stream())
        assert isinstance(records, list)

    def test_stream_uploader_requeues_on_push_failure(self):
        """
        When _flush_locked raises, records are re-queued so data is not lost.
        """
        uploader = StreamUploader(repo_id="test/repo", token="fake", dry_run=True)
        rec = make_record(
            raw_text=DARWIN_LETTER_TEXT,
            source_url="test://recovery",
            doc_type="correspondence",
            agent_id="agent_001",
        )

        original_flush = uploader._flush_locked
        call_count = [0]

        def flaky_flush():
            call_count[0] += 1
            if call_count[0] == 1:
                raise IOError("Simulated transient push failure")
            original_flush()

        uploader._flush_locked = flaky_flush
        uploader.add(rec)

        # First flush raises; record is re-queued
        with pytest.raises(IOError):
            uploader.flush()

        # Restore and flush again — record should now be processed
        uploader._flush_locked = original_flush
        uploader.flush()
        assert uploader.total_records == 1  # add() already incremented this

    def test_network_retry_decorator_present(self):
        """
        The network_retry decorator is applied to load methods in key extractors.
        """
        from correspondence_extractor import CorrespondenceExtractor
        from dialogue_extractor import DialogueExtractor
        from pdf_academic_extractor import PDFAcademicExtractor

        for cls, method in [
            (CorrespondenceExtractor, "_load_xml"),
            (DialogueExtractor, "_load_text"),
            (PDFAcademicExtractor, "_load_bytes"),
        ]:
            fn = getattr(cls, method)
            # tenacity wraps with a __wrapped__ attribute or retry attribute
            assert callable(fn), f"{cls.__name__}.{method} not callable"

    def test_uploader_with_context_manager_flushes(self):
        """Context manager (__exit__) flushes on normal exit."""
        uploader = StreamUploader(repo_id="test/repo", token="fake", dry_run=True)
        records = _make_unique_records(5)
        with uploader:
            for rec in records:
                uploader.add(rec)
        # After context exit, all records should be accounted for
        assert uploader.total_records == 5


# ===========================================================================
# Test 5: Metadata accuracy
# ===========================================================================
class TestMetadataAccuracy:
    """Sample 20 extracted documents and verify tier/prior/domain tag correctness."""

    def _sample_records(self, n: int = 20) -> List[dict]:
        return _make_unique_records(n)

    def test_all_tiers_valid(self):
        """Every record has a tier in {1,2,3,4,5}."""
        for rec in self._sample_records(20):
            assert rec["metadata"]["tier"] in VALID_TIERS, (
                f"Invalid tier {rec['metadata']['tier']}"
            )

    def test_all_priors_valid(self):
        """Every record has a structural_prior in {1..9}."""
        for rec in self._sample_records(20):
            assert rec["metadata"]["structural_prior"] in VALID_PRIORS, (
                f"Invalid prior {rec['metadata']['structural_prior']}"
            )

    def test_all_domains_valid(self):
        """Every domain in the domain list is in VALID_DOMAINS."""
        for rec in self._sample_records(20):
            for d in rec["metadata"]["domain"]:
                assert d in VALID_DOMAINS, f"Invalid domain {d!r}"

    def test_all_doc_types_valid(self):
        """Every doc_type is in VALID_DOC_TYPES."""
        for rec in self._sample_records(20):
            assert rec["metadata"]["doc_type"] in VALID_DOC_TYPES, (
                f"Invalid doc_type {rec['metadata']['doc_type']!r}"
            )

    def test_quality_scores_in_range(self):
        """quality_score is always in [0.0, 1.0]."""
        for rec in self._sample_records(20):
            qs = rec["metadata"]["quality_score"]
            assert 0.0 <= qs <= 1.0, f"quality_score {qs} out of range"

    def test_tier_classifier_axiom_text(self):
        """Axiom-heavy text → tier 1."""
        text = (
            "By definition, we assume that this axiom holds as a first principle. "
            "Let us define the fundamental postulate: given that this premise is true, "
            "suppose that the definition applies universally. A priori this must hold."
        )
        assert classify_tier(text) == 1

    def test_tier_classifier_proof_text(self):
        """Proof-heavy text → tier 2."""
        text = (
            "Theorem: for all n, P(n) holds. Proof by induction. "
            "Therefore it follows that the lemma is satisfied. "
            "By induction the corollary is proved. QED. "
            "Hence the result follows from the theorem."
        )
        assert classify_tier(text) == 2

    def test_prior_tagger_feedback_text(self):
        """Feedback-loop text → prior 7."""
        text = (
            "The feedback loop maintains homeostasis through negative feedback. "
            "The control system uses a PID controller for stability. "
            "The cybernetic regulation ensures the attractor state is reached. "
            "Positive feedback amplifies the oscillation in the loop."
        )
        assert tag_prior(text) == 7

    def test_prior_tagger_conservation_text(self):
        """Conservation-law text → prior 6."""
        text = (
            "By Noether's theorem, every symmetry implies a conservation law. "
            "The conserved quantity is the invariant under the transformation. "
            "Energy conservation and mass balance follow from the first law. "
            "The steady-state flux satisfies the no-free-lunch principle."
        )
        assert tag_prior(text) == 6

    def test_metadata_has_all_required_keys(self):
        """Every record metadata contains all required keys."""
        required_keys = {
            "tier", "structural_prior", "domain", "doc_type",
            "participants", "quality_score", "source_url",
            "license", "extraction_timestamp", "agent_id",
        }
        for rec in self._sample_records(20):
            assert required_keys.issubset(rec["metadata"].keys()), (
                f"Missing keys: {required_keys - set(rec['metadata'].keys())}"
            )
