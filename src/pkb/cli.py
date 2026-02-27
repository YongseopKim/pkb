"""PKB CLI entry point."""

from pathlib import Path

import click

from pkb import __version__


@click.group()
@click.version_option(version=__version__, prog_name="pkb")
@click.option("-v", "--verbose", count=True, help="Verbosity: -v=info, -vv=debug.")
def cli(verbose: int) -> None:
    """PKB — Private Knowledge Base CLI."""
    from pkb.logging_config import setup_logging
    setup_logging(verbosity=verbose)


@cli.command()
@click.option("--force", is_flag=True, help="Overwrite existing PKB home.")
def init(force: bool) -> None:
    """Initialize ~/.pkb/ directory structure."""
    from pkb.init import init_pkb_home

    try:
        pkb_home = init_pkb_home(force=force)
        click.echo(f"Initialized PKB home at {pkb_home}")
    except FileExistsError as e:
        raise click.ClickException(str(e))


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def parse(path: str) -> None:
    """Parse input file(s) and show summary.

    PATH can be a single .jsonl/.md file or a directory containing input files.
    """
    target = Path(path)

    if target.is_file():
        _parse_single_file(target)
    elif target.is_dir():
        _parse_directory(target)
    else:
        raise click.ClickException(f"Not a file or directory: {target}")


def _parse_single_file(path: Path) -> None:
    """Parse and summarize a single input file."""
    from pkb.parser.directory import parse_file

    conv = parse_file(path)
    click.echo(f"Platform: {conv.meta.platform}")
    click.echo(f"Title:    {conv.meta.title or '(untitled)'}")
    click.echo(f"Turns:    {conv.turn_count}")
    click.echo(f"Exported: {conv.meta.exported_at.isoformat()}")
    if conv.first_user_message:
        preview = conv.first_user_message[:80]
        if len(conv.first_user_message) > 80:
            preview += "..."
        click.echo(f"First message: {preview}")


def _parse_directory(directory: Path) -> None:
    """Parse and summarize all JSONL files in a directory."""
    from pkb.parser.directory import parse_directory

    conversations = parse_directory(directory)
    if not conversations:
        click.echo("No input files found.")
        return

    click.echo(f"Found {len(conversations)} conversation(s):\n")
    for conv in conversations:
        click.echo(f"  [{conv.meta.platform}] {conv.meta.title or '(untitled)'}")
        click.echo(f"    Turns: {conv.turn_count} | Exported: {conv.meta.exported_at.date()}")
    click.echo()
    platforms = {c.meta.platform for c in conversations}
    total_turns = sum(c.turn_count for c in conversations)
    platform_str = ", ".join(sorted(platforms))
    click.echo(f"Total: {len(conversations)} files, {total_turns} turns, platforms: {platform_str}")


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--kb", required=True, help="Knowledge base name (from config).")
@click.option("--dry-run", is_flag=True, help="Preview without writing files or DB.")
def ingest(path: str, kb: str, dry_run: bool) -> None:
    """Ingest input file(s) into a knowledge base.

    PATH can be a single .jsonl/.md file or a directory of input files.
    """
    from pkb.config import build_chunk_store, build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.generator.meta_gen import MetaGenerator
    from pkb.ingest import IngestPipeline, move_to_done
    from pkb.parser.directory import SUPPORTED_EXTENSIONS, find_input_files
    from pkb.vocab.loader import load_domains, load_topics

    target = Path(path)

    # Load config
    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    # Find KB
    kb_entry = None
    for entry in config.knowledge_bases:
        if entry.name == kb:
            kb_entry = entry
            break
    if kb_entry is None:
        raise click.ClickException(
            f"Knowledge base '{kb}' not found in config. "
            f"Available: {[e.name for e in config.knowledge_bases]}"
        )

    # Load vocab
    vocab_dir = pkb_home / "vocab"
    domains_vocab = load_domains(vocab_dir / "domains.yaml")
    topics_vocab = load_topics(vocab_dir / "topics.yaml")

    # Initialize services
    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        if dry_run:
            click.echo(f"[dry-run] DB connection skipped: {e}")
            repo = None  # type: ignore[assignment]
            chunk_store = None  # type: ignore[assignment]
        else:
            raise click.ClickException(f"Database connection failed: {e}")

    router = build_llm_router(config)
    meta_gen = MetaGenerator(config.meta_llm, router=router)

    pipeline = IngestPipeline(
        repo=repo,
        chunk_store=chunk_store,
        meta_gen=meta_gen,
        kb_path=kb_entry.path,
        kb_name=kb_entry.name,
        domains=list(domains_vocab.get_ids()),
        topics=list(topics_vocab.get_approved_canonicals()),
        dry_run=dry_run,
    )

    # Process file(s)
    input_files = []
    if target.is_file() and target.suffix in SUPPORTED_EXTENSIONS:
        input_files = [target]
    elif target.is_dir():
        input_files = find_input_files(target)
    else:
        raise click.ClickException(f"Not a supported file or directory: {target}")

    if not input_files:
        click.echo("No input files found.")
        return

    prefix = "[dry-run] " if dry_run else ""
    watch_dir = kb_entry.get_watch_dir()
    success = 0
    skipped = 0
    errors = 0

    merged = 0
    for f in input_files:
        try:
            result = pipeline.ingest_file(f)
            if result is None:
                skipped += 1
                click.echo(f"  {prefix}SKIP (duplicate): {f.name}")
            elif result.get("status", "").startswith("skip_"):
                skipped += 1
                reason = result.get("reason", "unknown")
                click.echo(f"  {prefix}SKIP ({reason}): {f.name}")
            elif result.get("merged"):
                merged += 1
                click.echo(
                    f"  {prefix}MERGE: {f.name} → {result['bundle_id']} ({result['platform']})"
                )
            else:
                success += 1
                click.echo(
                    f"  {prefix}OK: {f.name} → {result['bundle_id']}"
                )
            dest = move_to_done(f, watch_dir, dry_run=dry_run)
            if dest:
                click.echo(f"  {prefix}MOVED: {f.name} → .done/")
        except Exception as e:
            errors += 1
            click.echo(f"  ERROR: {f.name} — {e}")

    parts = [f"{success} ingested"]
    if merged:
        parts.append(f"{merged} merged")
    parts.extend([f"{skipped} skipped", f"{errors} errors"])
    click.echo(f"\n{prefix}Done: {', '.join(parts)}")

    # Cleanup
    if not dry_run and repo is not None:
        repo.close()


@cli.command()
@click.argument("source_dir", type=click.Path(exists=True))
@click.option("--kb", required=True, help="Knowledge base name (from config).")
@click.option("--no-resume", is_flag=True, help="Ignore checkpoint, process all files.")
@click.option("--max", "max_files", type=int, default=0, help="Max files to process (0=all).")
@click.option(
    "--workers", type=int, default=0,
    help="Concurrent workers (0=use config default). Enables concurrent mode.",
)
def batch(source_dir: str, kb: str, no_resume: bool, max_files: int, workers: int) -> None:
    """Batch process input files from a source directory.

    SOURCE_DIR is scanned recursively for .jsonl and .md files.
    Use --workers to enable concurrent processing.
    """
    from pkb.batch import BatchProcessor
    from pkb.config import build_chunk_store, build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.generator.meta_gen import MetaGenerator
    from pkb.ingest import IngestPipeline
    from pkb.vocab.loader import load_domains, load_topics

    target = Path(source_dir)
    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    kb_entry = None
    for entry in config.knowledge_bases:
        if entry.name == kb:
            kb_entry = entry
            break
    if kb_entry is None:
        raise click.ClickException(f"Knowledge base '{kb}' not found in config.")

    vocab_dir = pkb_home / "vocab"
    domains_vocab = load_domains(vocab_dir / "domains.yaml")
    topics_vocab = load_topics(vocab_dir / "topics.yaml")

    concurrency = config.concurrency
    if workers > 0:
        concurrency.max_concurrent_files = workers

    # Use connection pool if concurrent mode
    use_concurrent = workers > 0
    try:
        if use_concurrent:
            repo = BundleRepository.from_pool(config.database.postgres, concurrency)
        else:
            repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    router = build_llm_router(config)
    meta_gen = MetaGenerator(config.meta_llm, router=router)

    pipeline = IngestPipeline(
        repo=repo,
        chunk_store=chunk_store,
        meta_gen=meta_gen,
        kb_path=kb_entry.path,
        kb_name=kb_entry.name,
        domains=list(domains_vocab.get_ids()),
        topics=list(topics_vocab.get_approved_canonicals()),
    )

    # Build engine for concurrent mode
    watch_dir = kb_entry.get_watch_dir()
    engine = None
    if use_concurrent:
        from pkb.engine import IngestEngine
        from pkb.ingest import move_to_done

        def _ingest_with_move(path):
            result = pipeline.ingest_file(path)
            move_to_done(path, watch_dir)
            return result

        engine = IngestEngine(
            ingest_fn=_ingest_with_move,
            concurrency=concurrency,
        )

    checkpoint_path = pkb_home / f"batch-checkpoint-{kb}.yaml"
    processor = BatchProcessor(
        pipeline=pipeline,
        checkpoint_path=checkpoint_path,
        max_files=max_files,
        resume=not no_resume,
        engine=engine,
        watch_dir=watch_dir,
    )

    if use_concurrent:
        mode_str = f"concurrent ({concurrency.max_concurrent_files} workers)"
    else:
        mode_str = "sequential"
    click.echo(f"Scanning {target} for input files... ({mode_str})")
    stats = processor.process(target)
    click.echo(
        f"\nDone: {stats['success']} ingested, "
        f"{stats['skipped']} skipped, {stats['errors']} errors"
    )
    repo.close()


@cli.group(invoke_without_command=True)
@click.option(
    "--status",
    type=click.Choice(["pending", "approved", "merged", "all"]),
    default="all",
    help="Filter topics by status.",
)
@click.pass_context
def topics(ctx: click.Context, status: str) -> None:
    """List and manage L2 topic vocabulary."""
    if ctx.invoked_subcommand is not None:
        return

    _list_topics(status)


@topics.command("list")
@click.option(
    "--status",
    type=click.Choice(["pending", "approved", "merged", "all"]),
    default="all",
    help="Filter topics by status.",
)
def topics_list(status: str) -> None:
    """Explicitly list topics."""
    _list_topics(status)


def _list_topics(status: str) -> None:
    """Shared logic for listing topics."""
    from pkb.config import get_pkb_home
    from pkb.vocab.manager import TopicManager

    pkb_home = get_pkb_home()
    topics_path = pkb_home / "vocab" / "topics.yaml"
    if not topics_path.exists():
        raise click.ClickException(f"Topics file not found: {topics_path}")

    mgr = TopicManager(topics_path)
    filter_status = None if status == "all" else status
    topic_list = mgr.list_topics(status=filter_status)

    if not topic_list:
        click.echo(f"No topics with status '{status}'.")
        return

    click.echo(f"Topics ({status}):\n")
    for t in sorted(topic_list, key=lambda x: x.canonical):
        line = f"  {t.canonical}"
        if t.status != "approved":
            line += f"  [{t.status}]"
        if t.merged_into:
            line += f"  → {t.merged_into}"
        if t.aliases:
            line += f"  (aliases: {', '.join(t.aliases)})"
        click.echo(line)
    click.echo(f"\nTotal: {len(topic_list)} topics")


@topics.command("approve")
@click.argument("name")
def topics_approve(name: str) -> None:
    """Approve a pending topic."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.vocab.manager import TopicManager
    from pkb.vocab.syncer import TopicSyncer

    pkb_home = get_pkb_home()
    topics_path = pkb_home / "vocab" / "topics.yaml"
    if not topics_path.exists():
        raise click.ClickException(f"Topics file not found: {topics_path}")

    mgr = TopicManager(topics_path)

    # Check topic exists and is pending
    matching = [t for t in mgr.list_topics() if t.canonical == name]
    if not matching:
        raise click.ClickException(f"Topic '{name}' not found.")
    if matching[0].status != "pending":
        raise click.ClickException(f"Topic '{name}' is not pending (status: {matching[0].status}).")

    # Update YAML
    mgr.approve(name)

    # Sync to DB
    try:
        config = load_config(pkb_home / CONFIG_FILENAME)
        repo = BundleRepository(config.database.postgres)
        syncer = TopicSyncer(repo=repo)
        syncer.sync_approve(name)
        repo.close()
    except Exception as e:
        click.echo(f"  Warning: DB sync failed ({e}), YAML updated only.")

    click.echo(f"Approved: {name}")


@topics.command("merge")
@click.argument("name")
@click.option("--into", required=True, help="Target topic to merge into.")
def topics_merge(name: str, into: str) -> None:
    """Merge a topic into another topic."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.vocab.manager import TopicManager
    from pkb.vocab.syncer import TopicSyncer

    pkb_home = get_pkb_home()
    topics_path = pkb_home / "vocab" / "topics.yaml"
    if not topics_path.exists():
        raise click.ClickException(f"Topics file not found: {topics_path}")

    mgr = TopicManager(topics_path)

    # Check source exists
    matching = [t for t in mgr.list_topics() if t.canonical == name]
    if not matching:
        raise click.ClickException(f"Topic '{name}' not found.")

    # Check target exists
    target = [t for t in mgr.list_topics() if t.canonical == into]
    if not target:
        raise click.ClickException(f"Target topic '{into}' not found.")

    # Update YAML
    mgr.merge(name, into=into)

    # Sync to DB
    try:
        config = load_config(pkb_home / CONFIG_FILENAME)
        repo = BundleRepository(config.database.postgres)
        syncer = TopicSyncer(repo=repo)
        syncer.sync_merge(name, into=into)
        repo.close()
    except Exception as e:
        click.echo(f"  Warning: DB sync failed ({e}), YAML updated only.")

    click.echo(f"Merged: {name} → {into}")


@topics.command("reject")
@click.argument("name")
def topics_reject(name: str) -> None:
    """Reject and remove a topic."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.vocab.manager import TopicManager
    from pkb.vocab.syncer import TopicSyncer

    pkb_home = get_pkb_home()
    topics_path = pkb_home / "vocab" / "topics.yaml"
    if not topics_path.exists():
        raise click.ClickException(f"Topics file not found: {topics_path}")

    mgr = TopicManager(topics_path)

    # Check topic exists
    matching = [t for t in mgr.list_topics() if t.canonical == name]
    if not matching:
        raise click.ClickException(f"Topic '{name}' not found.")

    # Update YAML
    mgr.reject(name)

    # Sync to DB
    try:
        config = load_config(pkb_home / CONFIG_FILENAME)
        repo = BundleRepository(config.database.postgres)
        syncer = TopicSyncer(repo=repo)
        syncer.sync_reject(name)
        repo.close()
    except Exception as e:
        click.echo(f"  Warning: DB sync failed ({e}), YAML updated only.")

    click.echo(f"Rejected: {name}")


@cli.group(invoke_without_command=True)
@click.pass_context
def kb(ctx: click.Context) -> None:
    """List and manage knowledge bases."""
    if ctx.invoked_subcommand is not None:
        return
    _kb_list()


@kb.command("list")
def kb_list_cmd() -> None:
    """List configured knowledge bases with bundle counts."""
    _kb_list()


def _kb_list() -> None:
    """Shared logic for listing KBs."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    entries = config.knowledge_bases
    if not entries:
        click.echo("No knowledge bases configured.")
        return

    # Try to connect to DB for bundle counts
    repo = None
    try:
        from pkb.db.postgres import BundleRepository

        repo = BundleRepository(config.database.postgres)
    except Exception:
        pass

    click.echo("Knowledge Bases:\n")
    for entry in entries:
        if repo is not None:
            try:
                count = repo.count_by_kb(entry.name)
                count_str = str(count)
            except Exception:
                count_str = "?"
        else:
            count_str = "?"
        click.echo(f"  {entry.name:<20} {count_str:>5} bundles    {entry.path}")

    if repo is not None:
        repo.close()

    click.echo(f"\nTotal: {len(entries)} KB(s)")


@cli.group()
def dedup() -> None:
    """Duplicate bundle detection and management."""


@dedup.command("scan")
@click.option("--kb", default=None, help="Knowledge base name filter.")
def dedup_scan(kb: str | None) -> None:
    """Scan bundles for duplicates using embedding similarity."""
    from pkb.config import build_chunk_store, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.dedup import DuplicateDetector

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    detector = DuplicateDetector(
        repo=repo, chunk_store=chunk_store, config=config.dedup,
    )

    click.echo("Scanning for duplicates...")
    stats = detector.scan(kb=kb)
    click.echo(
        f"Done: {stats['scanned']} bundles scanned, "
        f"{stats['new_pairs']} new duplicate pairs found."
    )
    repo.close()


@dedup.command("list")
@click.option(
    "--status",
    type=click.Choice(["pending", "dismissed", "confirmed", "all"]),
    default="pending",
    help="Filter by status.",
)
def dedup_list(status: str) -> None:
    """List detected duplicate pairs."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.dedup import DuplicateDetector

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    detector = DuplicateDetector(
        repo=repo, chunk_store=None, config=config.dedup,  # type: ignore[arg-type]
    )

    filter_status = None if status == "all" else status
    pairs = detector.list_pairs(status=filter_status)

    if not pairs:
        click.echo(f"No duplicate pairs ({status}).")
    else:
        for p in pairs:
            click.echo(
                f"  [{p['id']}] {p['bundle_a']} ↔ {p['bundle_b']}  "
                f"(sim: {p['similarity']:.2f}, {p['status']})"
            )
        click.echo(f"\nTotal: {len(pairs)} pair(s)")
    repo.close()


@dedup.command("dismiss")
@click.argument("pair_id", type=int)
def dedup_dismiss(pair_id: int) -> None:
    """Dismiss a duplicate pair (mark as non-duplicate)."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.dedup import DuplicateDetector

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    detector = DuplicateDetector(
        repo=repo, chunk_store=None, config=config.dedup,  # type: ignore[arg-type]
    )
    detector.dismiss_pair(pair_id)
    click.echo(f"Dismissed pair #{pair_id}")
    repo.close()


@dedup.command("confirm")
@click.argument("pair_id", type=int)
def dedup_confirm(pair_id: int) -> None:
    """Confirm a duplicate pair."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.dedup import DuplicateDetector

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    detector = DuplicateDetector(
        repo=repo, chunk_store=None, config=config.dedup,  # type: ignore[arg-type]
    )
    detector.confirm_pair(pair_id)
    click.echo(f"Confirmed pair #{pair_id}")
    repo.close()


@cli.group()
def relate() -> None:
    """Knowledge graph: discover and manage bundle relations."""


@relate.command("scan")
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option(
    "--type",
    "relation_type",
    type=click.Choice(["similar", "related", "all"]),
    default="all",
    help="Relation type to scan for.",
)
def relate_scan(kb: str | None, relation_type: str) -> None:
    """Scan bundles to discover relations (similar/related)."""
    from pkb.config import build_chunk_store, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.relations import RelationBuilder

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    builder = RelationBuilder(
        repo=repo, chunk_store=chunk_store, config=config.relations,
    )

    click.echo("Scanning for relations...")
    stats = builder.scan(kb=kb)
    click.echo(
        f"Done: {stats['scanned']} bundles scanned, "
        f"{stats['new_relations']} relations found."
    )
    repo.close()


@relate.command("list")
@click.option(
    "--type",
    "relation_type",
    type=click.Choice(["similar", "related", "all"]),
    default="all",
    help="Filter by relation type.",
)
@click.option("--kb", default=None, help="Knowledge base name filter.")
def relate_list(relation_type: str, kb: str | None) -> None:
    """List all discovered relations."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    filter_type = None if relation_type == "all" else relation_type
    relations = repo.list_all_relations(relation_type=filter_type, kb=kb)

    if not relations:
        click.echo("No relations found.")
    else:
        for r in relations:
            click.echo(
                f"  {r['source_bundle_id']} → {r['target_bundle_id']}  "
                f"({r['relation_type']}, score: {r['score']:.2f})"
            )
        click.echo(f"\nTotal: {len(relations)} relation(s)")
    repo.close()


@relate.command("show")
@click.argument("bundle_id")
@click.option(
    "--type",
    "relation_type",
    type=click.Choice(["similar", "related", "all"]),
    default="all",
    help="Filter by relation type.",
)
def relate_show(bundle_id: str, relation_type: str) -> None:
    """Show relations for a specific bundle."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    filter_type = None if relation_type == "all" else relation_type
    relations = repo.list_relations(bundle_id, relation_type=filter_type)

    if not relations:
        click.echo(f"No relations for {bundle_id}")
    else:
        click.echo(f"Relations for {bundle_id}:")
        for r in relations:
            other = (
                r["target_bundle_id"]
                if r["source_bundle_id"] == bundle_id
                else r["source_bundle_id"]
            )
            click.echo(
                f"  → {other}  ({r['relation_type']}, score: {r['score']:.2f})"
            )
        click.echo(f"\nTotal: {len(relations)} relation(s)")
    repo.close()


@cli.command()
@click.argument("query")
@click.option(
    "--mode", type=click.Choice(["hybrid", "keyword", "semantic"]),
    default="hybrid", help="Search mode.",
)
@click.option("--domain", "domains", multiple=True, help="Domain filter (repeatable).")
@click.option("--topic", "topics", multiple=True, help="Topic filter (repeatable).")
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option("--after", default=None, help="Date filter start (YYYY-MM-DD).")
@click.option("--before", default=None, help="Date filter end (YYYY-MM-DD).")
@click.option("--limit", default=10, type=int, help="Max results (default: 10).")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def search(
    query: str, mode: str, domains: tuple, topics: tuple,
    kb: str | None, after: str | None, before: str | None,
    limit: int, as_json: bool,
) -> None:
    """Search bundles across knowledge bases.

    QUERY is the search text.
    """
    import json as json_mod
    from datetime import date as date_cls

    from pkb.config import build_chunk_store, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.search.engine import SearchEngine
    from pkb.search.models import SearchMode, SearchQuery

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    engine = SearchEngine(repo=repo, chunk_store=chunk_store)

    # Parse date filters
    after_date = date_cls.fromisoformat(after) if after else None
    before_date = date_cls.fromisoformat(before) if before else None

    search_query = SearchQuery(
        query=query,
        mode=SearchMode(mode),
        domains=list(domains),
        topics=list(topics),
        kb=kb,
        after=after_date,
        before=before_date,
        limit=limit,
    )

    results = engine.search(search_query)

    if as_json:
        output = [
            {
                "bundle_id": r.bundle_id,
                "summary": r.summary,
                "domains": r.domains,
                "topics": r.topics,
                "score": r.score,
                "created_at": r.created_at.isoformat(),
                "source": r.source,
            }
            for r in results
        ]
        click.echo(json_mod.dumps(output, ensure_ascii=False, indent=2))
    elif not results:
        click.echo("No results found.")
    else:
        for i, r in enumerate(results, 1):
            domain_str = ", ".join(r.domains) if r.domains else "-"
            topic_str = ", ".join(r.topics) if r.topics else "-"
            click.echo(f"[{i}] {r.bundle_id}  (score: {r.score:.2f}, {r.source})")
            if r.summary:
                summary_preview = r.summary[:100]
                if len(r.summary) > 100:
                    summary_preview += "..."
                click.echo(f"    {summary_preview}")
            click.echo(f"    Domains: {domain_str} | Topics: {topic_str}")
            click.echo()
        click.echo(f"Found {len(results)} result(s).")

    repo.close()


@cli.command()
@click.argument("bundle_id", required=False, default=None)
@click.option("--kb", required=True, help="Knowledge base name (from config).")
@click.option("--full", is_flag=True, help="Reindex all bundles + remove orphan DB records.")
def reindex(bundle_id: str | None, kb: str, full: bool) -> None:
    """Sync _bundle.md edits to DB.

    Reindex a single BUNDLE_ID or use --full to reindex everything.
    """
    from pkb.config import build_chunk_store, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.reindex import Reindexer

    if not bundle_id and not full:
        raise click.ClickException("Provide a BUNDLE_ID or use --full.")

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    kb_entry = None
    for entry in config.knowledge_bases:
        if entry.name == kb:
            kb_entry = entry
            break
    if kb_entry is None:
        raise click.ClickException(
            f"Knowledge base '{kb}' not found in config. "
            f"Available: {[e.name for e in config.knowledge_bases]}"
        )

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    reindexer = Reindexer(
        repo=repo,
        chunk_store=chunk_store,
        kb_path=kb_entry.path,
        kb_name=kb_entry.name,
        embedding_config=config.embedding,
    )

    if full:
        def _progress(bid, status):
            click.echo(f"  [{status}] {bid}")

        stats = reindexer.reindex_full(progress_callback=_progress)
        click.echo(
            f"\nDone: {stats['updated']} updated, {stats['skipped']} skipped, "
            f"{stats['errors']} errors, {stats['deleted']} deleted "
            f"(total: {stats['total']})"
        )
    else:
        result = reindexer.reindex_bundle(bundle_id)
        click.echo(f"[{result['status']}] {result['bundle_id']}")
        if "reason" in result:
            click.echo(f"  Reason: {result['reason']}")

    repo.close()


@cli.command()
@click.argument("bundle_id", required=False, default=None)
@click.option("--kb", required=True, help="Knowledge base name (from config).")
@click.option("--all", "all_bundles", is_flag=True, help="Regenerate all bundles.")
@click.option("--dry-run", is_flag=True, help="Preview without writing DB.")
def regenerate(bundle_id: str | None, kb: str, all_bundles: bool, dry_run: bool) -> None:
    """Regenerate derived files from raw JSONL.

    Re-runs LLM meta extraction with current prompts/model.
    Provide BUNDLE_ID for a single bundle, or use --all.
    """
    from pkb.config import build_chunk_store, build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.generator.meta_gen import MetaGenerator
    from pkb.regenerate import Regenerator
    from pkb.vocab.loader import load_domains, load_topics

    if not bundle_id and not all_bundles:
        raise click.ClickException("Provide a BUNDLE_ID or use --all.")

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    kb_entry = None
    for entry in config.knowledge_bases:
        if entry.name == kb:
            kb_entry = entry
            break
    if kb_entry is None:
        raise click.ClickException(
            f"Knowledge base '{kb}' not found in config. "
            f"Available: {[e.name for e in config.knowledge_bases]}"
        )

    vocab_dir = pkb_home / "vocab"
    domains_vocab = load_domains(vocab_dir / "domains.yaml")
    topics_vocab = load_topics(vocab_dir / "topics.yaml")

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        if dry_run:
            click.echo(f"[dry-run] DB connection skipped: {e}")
            repo = None  # type: ignore[assignment]
            chunk_store = None  # type: ignore[assignment]
        else:
            raise click.ClickException(f"Database connection failed: {e}")

    router = build_llm_router(config)
    meta_gen = MetaGenerator(config.meta_llm, router=router)

    regen = Regenerator(
        repo=repo,
        chunk_store=chunk_store,
        meta_gen=meta_gen,
        kb_path=kb_entry.path,
        kb_name=kb_entry.name,
        domains=list(domains_vocab.get_ids()),
        topics=list(topics_vocab.get_approved_canonicals()),
        dry_run=dry_run,
    )

    prefix = "[dry-run] " if dry_run else ""

    if all_bundles:
        def _progress(bid, status):
            click.echo(f"  {prefix}[{status}] {bid}")

        stats = regen.regenerate_all(progress_callback=_progress)
        click.echo(
            f"\n{prefix}Done: {stats['regenerated']} regenerated, "
            f"{stats['errors']} errors (total: {stats['total']})"
        )
    else:
        result = regen.regenerate_bundle(bundle_id)
        click.echo(f"{prefix}[{result['status']}] {result['bundle_id']}")
        if "reason" in result:
            click.echo(f"  Reason: {result['reason']}")

    if not dry_run and repo is not None:
        repo.close()


def _find_watch_dir_for_path(
    file_path: Path, resolved_map: dict[str, object],
) -> tuple[str, object] | None:
    """Find the watch_dir that contains file_path (supports subdirectories).

    Returns (watch_dir_str, value) or None if no match.
    """
    resolved = file_path.resolve()
    for watch_dir_str, value in resolved_map.items():
        if resolved.is_relative_to(Path(watch_dir_str)):
            return watch_dir_str, value
    return None


def _build_watch_callback(
    *,
    pipelines: dict,
    kb_entries: dict,
    repo,
    chunk_store,
):
    """Build the callback for KBWatcher that handles new/modified files.

    When a file was previously ingested (tracked via source_path), the old bundle
    is completely removed (DB + ChromaDB + filesystem) before re-ingesting.
    """
    import shutil

    from pkb.ingest import move_to_done

    # Build resolved-path lookup to handle symlinks (e.g. macOS /tmp -> /private/tmp)
    resolved_pipelines = {str(Path(k).resolve()): v for k, v in pipelines.items()}
    resolved_kb_entries = {str(Path(k).resolve()): v for k, v in kb_entries.items()}

    def _on_new_file(file_path):
        match = _find_watch_dir_for_path(file_path, resolved_pipelines)
        if match is None:
            click.echo(f"  WARNING: No pipeline for {file_path}")
            return
        _, pipeline = match

        kb_match = _find_watch_dir_for_path(file_path, resolved_kb_entries)
        kb_entry = kb_match[1] if kb_match else None

        # Check if this file was previously ingested
        existing_bundle_id = repo.find_by_source_path(str(file_path))

        if existing_bundle_id:
            # Delete old bundle completely (DB + ChromaDB + filesystem)
            chunk_store.delete_by_bundle(existing_bundle_id)
            if kb_entry:
                bundle_dir = kb_entry.path / "bundles" / existing_bundle_id
                shutil.rmtree(bundle_dir, ignore_errors=True)
            repo.delete_bundle(existing_bundle_id)
            click.echo(f"  REINGEST: {file_path.name} (deleted {existing_bundle_id})")

        try:
            result = pipeline.ingest_file(file_path, force=bool(existing_bundle_id))
            if result is None:
                click.echo(f"  SKIP (duplicate): {file_path.name}")
            elif result.get("status", "").startswith("skip_"):
                reason = result.get("reason", "unknown")
                click.echo(f"  SKIP ({reason}): {file_path.name}")
            elif result.get("merged"):
                click.echo(
                    f"  MERGE: {file_path.name} → {result['bundle_id']}"
                    f" ({result['platform']})"
                )
            else:
                click.echo(f"  OK: {file_path.name} -> {result['bundle_id']}")
            if kb_entry:
                dest = move_to_done(file_path, kb_entry.get_watch_dir())
                if dest:
                    click.echo(f"  MOVED: {file_path.name} → .done/")
        except Exception as e:
            click.echo(f"  ERROR: {file_path.name} — {e}")

    return _on_new_file


def _build_watch_ingest_fn(
    *,
    pipelines: dict,
    kb_entries: dict,
    repo,
    chunk_store,
):
    """Build an ingest_fn callable for IngestEngine in watch mode.

    Handles reingest logic: if file was previously ingested (tracked via source_path),
    the old bundle is completely removed before re-ingesting.
    """
    import shutil

    from pkb.ingest import move_to_done

    # Build resolved-path lookup to handle symlinks (e.g. macOS /tmp -> /private/tmp)
    resolved_pipelines = {str(Path(k).resolve()): v for k, v in pipelines.items()}
    resolved_kb_entries = {str(Path(k).resolve()): v for k, v in kb_entries.items()}

    def _ingest_fn(file_path):
        match = _find_watch_dir_for_path(file_path, resolved_pipelines)
        if match is None:
            return None
        _, pipeline = match

        kb_match = _find_watch_dir_for_path(file_path, resolved_kb_entries)
        kb_entry = kb_match[1] if kb_match else None

        # Check if this file was previously ingested
        existing_bundle_id = repo.find_by_source_path(str(file_path))

        if existing_bundle_id:
            # Delete old bundle completely (DB + ChromaDB + filesystem)
            chunk_store.delete_by_bundle(existing_bundle_id)
            if kb_entry:
                bundle_dir = kb_entry.path / "bundles" / existing_bundle_id
                shutil.rmtree(bundle_dir, ignore_errors=True)
            repo.delete_bundle(existing_bundle_id)

        result = pipeline.ingest_file(file_path, force=bool(existing_bundle_id))
        if kb_entry:
            move_to_done(file_path, kb_entry.get_watch_dir())
        return result

    return _ingest_fn


async def _initial_scan(
    watch_dirs: list[Path], collector,
) -> int:
    """Scan existing files in watch directories and feed to EventCollector.

    Called at watch startup to process files that arrived before watch started.
    """
    from pkb.parser.directory import find_input_files_recursive

    count = 0
    for d in watch_dirs:
        for f in find_input_files_recursive(d):
            await collector.put(f)
            count += 1
    return count


async def _periodic_retry_scan(
    watch_dirs: list[Path],
    collector,
    shutdown_event,
    interval_seconds: float = 300.0,
) -> None:
    """Periodically rescan inbox directories and re-queue failed files.

    Files that failed on previous attempts remain in inbox (not moved to .done/).
    EventCollector's internal dedup prevents re-queuing files currently being processed.
    """
    import asyncio

    from pkb.parser.directory import find_input_files_recursive

    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=interval_seconds)
            return  # shutdown requested
        except asyncio.TimeoutError:
            pass  # interval elapsed, do rescan

        count = 0
        for d in watch_dirs:
            for f in find_input_files_recursive(d):
                await collector.put(f)
                count += 1
        if count > 0:
            click.echo(f"Retry scan: {count} file(s) re-queued.")


@cli.command()
@click.option("--kb", default=None, help="Watch specific KB only (default: all KBs).")
def watch(kb: str | None) -> None:
    """Watch KB inbox directories for new input files and auto-ingest.

    Uses concurrent ingest engine for parallel processing.
    Runs as a foreground process. Press Ctrl+C to stop.
    """
    import asyncio
    import signal

    from pkb.config import build_chunk_store, build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.engine import EventCollector, IngestEngine
    from pkb.generator.meta_gen import MetaGenerator
    from pkb.ingest import IngestPipeline
    from pkb.vocab.loader import load_domains, load_topics
    from pkb.watcher import AsyncFileEventHandler

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    # Filter KB entries
    entries = config.knowledge_bases
    if kb:
        entries = [e for e in entries if e.name == kb]
        if not entries:
            raise click.ClickException(
                f"Knowledge base '{kb}' not found in config. "
                f"Available: {[e.name for e in config.knowledge_bases]}"
            )

    if not entries:
        raise click.ClickException("No knowledge bases configured.")

    # Load vocab
    vocab_dir = pkb_home / "vocab"
    domains_vocab = load_domains(vocab_dir / "domains.yaml")
    topics_vocab = load_topics(vocab_dir / "topics.yaml")

    concurrency = config.concurrency

    # Use connection pool for concurrent watch mode
    try:
        repo = BundleRepository.from_pool(config.database.postgres, concurrency)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    router = build_llm_router(config)
    meta_gen = MetaGenerator(config.meta_llm, router=router)

    # Build pipelines + kb_entries per KB
    pipelines: dict[str, IngestPipeline] = {}
    kb_entries_map: dict[str, object] = {}
    watch_dirs = []
    for entry in entries:
        watch_dir = entry.get_watch_dir()
        watch_dir.mkdir(parents=True, exist_ok=True)
        watch_dir_str = str(watch_dir)
        pipelines[watch_dir_str] = IngestPipeline(
            repo=repo,
            chunk_store=chunk_store,
            meta_gen=meta_gen,
            kb_path=entry.path,
            kb_name=entry.name,
            domains=list(domains_vocab.get_ids()),
            topics=list(topics_vocab.get_approved_canonicals()),
        )
        kb_entries_map[watch_dir_str] = entry
        watch_dirs.append(watch_dir)

    ingest_fn = _build_watch_ingest_fn(
        pipelines=pipelines,
        kb_entries=kb_entries_map,
        repo=repo,
        chunk_store=chunk_store,
    )

    def _progress(result):
        if result.status == "merged":
            click.echo(
                f"  MERGE: {result.path.name} → {result.bundle_id}"
                f" ({result.platform})"
            )
        elif result.status == "ok":
            click.echo(f"  OK: {result.path.name} → {result.bundle_id}")
        elif result.status == "skipped":
            click.echo(f"  SKIP: {result.path.name}")
        elif result.status == "error":
            click.echo(f"  ERROR: {result.path.name} — {result.error}")

    engine = IngestEngine(
        ingest_fn=ingest_fn,
        concurrency=concurrency,
        progress_callback=_progress,
    )
    collector = EventCollector(concurrency)

    dirs_str = ", ".join(str(d) for d in watch_dirs)
    click.echo(f"Watching: {dirs_str}")
    click.echo(f"Concurrent workers: {concurrency.max_concurrent_files}")
    click.echo("Press Ctrl+C to stop.\n")

    async def _run():
        from watchdog.observers import Observer

        loop = asyncio.get_running_loop()
        shutdown = asyncio.Event()

        handler = AsyncFileEventHandler(loop=loop, collector=collector)
        observer = Observer()
        for d in watch_dirs:
            observer.schedule(handler, str(d), recursive=True)
        observer.start()

        # Scan existing files in inbox (arrived before watch started)
        scan_count = await _initial_scan(watch_dirs, collector)
        if scan_count > 0:
            click.echo(f"Initial scan: {scan_count} existing file(s) queued.")

        # Start periodic retry scan (re-queues failed files)
        retry_task = None
        if concurrency.retry_interval > 0:
            retry_task = asyncio.create_task(
                _periodic_retry_scan(
                    watch_dirs, collector, shutdown,
                    interval_seconds=concurrency.retry_interval,
                )
            )

        # Handle signals
        def _signal_handler():
            shutdown.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, _signal_handler)
            except NotImplementedError:
                pass  # Windows

        try:
            await engine.run_watch(collector, shutdown_event=shutdown)
        finally:
            if retry_task is not None:
                retry_task.cancel()
            observer.stop()
            observer.join()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        repo.close()
        click.echo("\nStopped watching.")


@cli.command()
@click.option("--topic", default=None, help="Topic to digest.")
@click.option("--domain", default=None, help="Domain to digest.")
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save to file.")
def digest(topic: str | None, domain: str | None, kb: str | None, output: str | None) -> None:
    """Generate a knowledge digest for a topic or domain."""
    if not topic and not domain:
        raise click.ClickException("Specify --topic or --domain")

    from pkb.config import build_chunk_store, build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.digest import DigestEngine
    from pkb.search.engine import SearchEngine

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    search_engine = SearchEngine(repo=repo, chunk_store=chunk_store)
    router = build_llm_router(config)

    engine = DigestEngine(
        repo=repo,
        search_engine=search_engine,
        router=router,
        config=config.digest,
    )

    click.echo("Generating digest...")

    if topic:
        result = engine.digest_topic(topic, kb=kb)
    else:
        result = engine.digest_domain(domain, kb=kb)

    if output:
        from pathlib import Path

        Path(output).write_text(result.content, encoding="utf-8")
        click.echo(f"Saved to {output}")
    else:
        click.echo(f"\n{result.content}")
        click.echo(f"\n--- {result.bundle_count} bundles referenced ---")

    repo.close()


@cli.command()
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option(
    "--mode",
    type=click.Choice(["explorer", "analyst", "writer"]),
    default="explorer",
    help="Conversation mode.",
)
def chat(kb: str | None, mode: str) -> None:
    """Interactive RAG chatbot.

    Ask questions against your knowledge base. Type 'quit' or 'exit' to stop.
    """
    from pkb.chat.engine import ChatEngine
    from pkb.chat.models import ChatSession
    from pkb.config import build_chunk_store, build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.search.engine import SearchEngine

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    search_engine = SearchEngine(repo=repo, chunk_store=chunk_store)
    router = build_llm_router(config)

    engine = ChatEngine(
        search_engine=search_engine,
        router=router,
        kb=kb,
        mode=mode,
    )

    session = ChatSession()

    click.echo("PKB RAG Chat (type 'quit' to exit)\n")

    while True:
        try:
            user_input = click.prompt("You", prompt_suffix="> ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("quit", "exit", "q"):
            break

        try:
            response = engine.ask(user_input, session=session)
            click.echo(f"\nAssistant> {response.content}\n")
        except Exception as e:
            click.echo(f"\nError: {e}\n")

    repo.close()
    click.echo("Bye!")


@cli.command("mcp-serve")
def mcp_serve() -> None:
    """Start PKB as an MCP server (stdio transport)."""
    from pkb.mcp_server import main

    main()


@cli.command()
@click.option("--port", default=8080, type=int, help="Server port (default: 8080).")
@click.option("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1).")
def web(port: int, host: str) -> None:
    """Launch local web UI.

    Starts a FastAPI server with bundle management, search, and topic management.
    """
    import uvicorn

    from pkb.chat.engine import ChatEngine
    from pkb.config import build_chunk_store, build_llm_router, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.search.engine import SearchEngine
    from pkb.web.app import create_app
    from pkb.web.deps import AppState

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    search_engine = SearchEngine(repo=repo, chunk_store=chunk_store)
    router = build_llm_router(config)
    chat_engine = ChatEngine(search_engine=search_engine, router=router)

    state = AppState(
        repo=repo,
        chunk_store=chunk_store,
        search_engine=search_engine,
        chat_engine=chat_engine,
    )

    app = create_app(state)
    click.echo(f"Starting PKB Web UI at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option("--skip-llm", is_flag=True, help="Skip LLM provider checks.")
@click.option("--skip-db", is_flag=True, help="Skip database checks.")
def doctor(skip_llm: bool, skip_db: bool) -> None:
    """Diagnose PKB configuration and connectivity."""
    from pkb.config import get_pkb_home
    from pkb.doctor import DoctorRunner, format_results

    pkb_home = get_pkb_home()
    runner = DoctorRunner(pkb_home=pkb_home)
    sections = runner.run_all_sectioned(skip_db=skip_db, skip_llm=skip_llm)
    click.echo(format_results(sections))


def _load_dsn() -> str:
    """Load PostgreSQL DSN from PKB config.yaml."""
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)
    return config.database.postgres.get_dsn()


@cli.group()
def db() -> None:
    """Database migration management (Alembic)."""


@db.command()
@click.option("--revision", default="head", help="Target revision (default: head).")
def upgrade(revision: str) -> None:
    """Upgrade database schema to a revision."""
    from pkb.db.migration_runner import run_upgrade

    dsn = _load_dsn()
    run_upgrade(dsn, revision=revision)
    click.echo(f"Upgraded to {revision}.")


@db.command()
@click.argument("revision")
def downgrade(revision: str) -> None:
    """Downgrade database schema to a revision."""
    from pkb.db.migration_runner import run_downgrade

    dsn = _load_dsn()
    run_downgrade(dsn, revision=revision)
    click.echo(f"Downgraded to {revision}.")


@db.command()
def current() -> None:
    """Show current database revision."""
    from pkb.db.migration_runner import get_current

    dsn = _load_dsn()
    get_current(dsn)


@db.command()
def history() -> None:
    """Show migration history."""
    from pkb.db.migration_runner import get_history

    dsn = _load_dsn()
    get_history(dsn)


@db.command()
@click.argument("revision")
def stamp(revision: str) -> None:
    """Stamp database with a revision without running migrations."""
    from pkb.db.migration_runner import run_stamp

    dsn = _load_dsn()
    run_stamp(dsn, revision=revision)
    click.echo(f"Stamped at {revision}.")


@db.command()
@click.option("--kb", required=True, help="Knowledge base name to reset.")
def reset(kb: str) -> None:
    """Delete ALL data for a KB (PostgreSQL + ChromaDB + filesystem).

    Requires typing the KB name to confirm. This is irreversible.
    """
    import shutil

    from pkb.config import build_chunk_store, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    # Validate KB exists
    kb_entry = None
    for entry in config.knowledge_bases:
        if entry.name == kb:
            kb_entry = entry
            break
    if kb_entry is None:
        raise click.ClickException(
            f"Knowledge base '{kb}' not found in config. "
            f"Available: {[e.name for e in config.knowledge_bases]}"
        )

    # Confirmation prompt
    confirm = click.prompt(
        f"Type '{kb}' to confirm deletion of ALL data for this KB"
    )
    if confirm != kb:
        click.echo("Aborted.")
        return

    # Connect to DBs
    try:
        repo = BundleRepository(config.database.postgres)
        chunk_store = build_chunk_store(config)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    # 1. PostgreSQL: delete all bundles (CASCADE)
    deleted_count = repo.delete_by_kb(kb)
    click.echo(f"PostgreSQL: {deleted_count} bundle(s) deleted")

    # 2. ChromaDB: delete all chunks
    chunk_store.delete_by_kb(kb)
    click.echo("ChromaDB: chunks deleted")

    # 3. Filesystem: clear bundles/ and inbox/.done/
    bundles_dir = kb_entry.path / "bundles"
    if bundles_dir.exists():
        shutil.rmtree(bundles_dir)
        bundles_dir.mkdir()
        click.echo(f"Filesystem: {bundles_dir} cleared")

    done_dir = kb_entry.get_watch_dir() / ".done"
    if done_dir.exists():
        shutil.rmtree(done_dir)
        click.echo(f"Filesystem: {done_dir} cleared")

    repo.close()
    click.echo(f"\nReset complete for KB '{kb}'.")


@db.command("migrate-domain")
@click.argument("old_domain")
@click.argument("new_domain")
def migrate_domain(old_domain: str, new_domain: str) -> None:
    """Rename a domain in bundle_domains (e.g., coding → dev).

    Bundles that already have NEW_DOMAIN will have OLD_DOMAIN removed
    instead of duplicated.
    """
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    count = repo.rename_domain(old_domain, new_domain)
    click.echo(f"Migrated {count} bundle(s): '{old_domain}' → '{new_domain}'")
    repo.close()


@cli.command()
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option("--domain", is_flag=True, default=False, help="Show domain distribution detail.")
@click.option("--json", "json_mode", is_flag=True, default=False, help="JSON output.")
def stats(kb: str | None, domain: bool, json_mode: bool) -> None:
    """Show knowledge base statistics.

    Displays overview stats by default. Use --domain for domain breakdown,
    --json for machine-readable output.
    """
    from pkb.analytics import AnalyticsEngine
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    engine = AnalyticsEngine(repo=repo)

    if domain:
        data = engine.domain_distribution(kb=kb)
        if json_mode:
            import json

            click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            click.echo("Domain Distribution:")
            for d in data:
                click.echo(f"  {d['domain']}: {d['count']}")
    else:
        data = engine.overview(kb=kb)
        if json_mode:
            import json

            click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            click.echo(f"Total bundles:   {data['total_bundles']}")
            click.echo(f"Total relations: {data['total_relations']}")
            click.echo(f"Domains:         {data['domain_count']}")
            click.echo(f"Topics:          {data['topic_count']}")

    repo.close()


@cli.command()
@click.option(
    "--period",
    type=click.Choice(["weekly", "monthly"]),
    default="weekly",
    help="Report period (default: weekly).",
)
@click.option("--kb", default=None, help="Knowledge base name filter.")
@click.option("--output", "-o", default=None, type=click.Path(), help="Save to file.")
def report(period: str, kb: str | None, output: str | None) -> None:
    """Generate a knowledge activity report.

    Produces a markdown report summarizing recent activity.
    """
    from pkb.analytics import AnalyticsEngine
    from pkb.config import get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.report import ReportGenerator

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    try:
        repo = BundleRepository(config.database.postgres)
    except Exception as e:
        raise click.ClickException(f"Database connection failed: {e}")

    engine = AnalyticsEngine(repo=repo)
    gen = ReportGenerator(repo=repo, analytics=engine)

    if period == "monthly":
        content = gen.monthly(kb=kb)
    else:
        content = gen.weekly(kb=kb)

    if output:
        from pathlib import Path

        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Saved to {output}")
    else:
        click.echo(content)

    repo.close()


@cli.command()
@click.argument("bundle_id", required=False, default=None)
@click.option("--kb", required=True, help="Knowledge base name (from config).")
@click.option("--all", "all_bundles", is_flag=True, help="Re-embed all bundles.")
@click.option(
    "--fresh", is_flag=True,
    help="Drop and recreate collection before re-embedding (use with --all).",
)
def reembed(
    bundle_id: str | None, kb: str, all_bundles: bool, fresh: bool,
) -> None:
    """Re-embed bundles with current embedding model.

    Use after changing the embedding model in config.yaml.
    Provide BUNDLE_ID for a single bundle, or use --all.
    --fresh drops the collection and recreates it (recommended for model changes).
    """
    from pkb.config import build_chunk_store, get_pkb_home, load_config
    from pkb.constants import CONFIG_FILENAME
    from pkb.db.postgres import BundleRepository
    from pkb.reembed import ReembedEngine

    if not bundle_id and not all_bundles:
        raise click.ClickException("Provide a BUNDLE_ID or use --all.")

    if fresh and not all_bundles:
        raise click.ClickException("--fresh can only be used with --all.")

    pkb_home = get_pkb_home()
    config = load_config(pkb_home / CONFIG_FILENAME)

    kb_entry = None
    for entry in config.knowledge_bases:
        if entry.name == kb:
            kb_entry = entry
            break
    if kb_entry is None:
        raise click.ClickException(
            f"Knowledge base '{kb}' not found in config. "
            f"Available: {[e.name for e in config.knowledge_bases]}"
        )

    repo = BundleRepository(config.database.postgres)
    chunk_store = build_chunk_store(config)

    engine = ReembedEngine(
        kb_path=kb_entry.path,
        kb_name=kb_entry.name,
        chunk_store=chunk_store,
        repo=repo,
        chunk_size=config.embedding.chunk_size,
        chunk_overlap=config.embedding.chunk_overlap,
    )

    if fresh:
        confirm = click.prompt(
            f"This will DELETE and recreate the ChromaDB collection. "
            f"Type the KB name '{kb}' to confirm"
        )
        if confirm != kb:
            raise click.ClickException("Confirmation mismatch — aborted.")

        def _progress(bid, status):
            click.echo(f"  [{status}] {bid}")

        stats = engine.reembed_collection_fresh(progress_callback=_progress)
        click.echo(
            f"\nDone (fresh): {stats['reembedded']} reembedded, "
            f"{stats['errors']} errors (total: {stats['total']})"
        )
    elif all_bundles:
        def _progress(bid, status):
            click.echo(f"  [{status}] {bid}")

        stats = engine.reembed_all(progress_callback=_progress)
        click.echo(
            f"\nDone: {stats['reembedded']} reembedded, "
            f"{stats['errors']} errors (total: {stats['total']})"
        )
    else:
        result = engine.reembed_bundle(bundle_id)
        if result["status"] == "reembedded":
            click.echo(
                f"Reembedded {result['bundle_id']} ({result.get('chunks', 0)} chunks)"
            )
        else:
            raise click.ClickException(
                f"Failed: {result.get('error', 'unknown error')}"
            )

    repo.close()


if __name__ == "__main__":
    cli()
