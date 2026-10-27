"""End-to-end test of the autonomous ML experimentation agent engine."""

import time
from pathlib import Path
from database import init_db, session_scope
from models import Dataset, ExperimentSession, Experiment, AgentDecision
from services import dataset_analyzer as da
from services import experiment_manager as em

def main():
    print("1. Initializing DB...")
    init_db()

    samples_dir = Path("sample_data")
    sample_file = samples_dir / "sample_linear_regression.csv"
    target = "price"

    print(f"2. Loading {sample_file}...")
    frame = da.load_dataframe(sample_file)
    analysis = da.analyze_dataframe(frame, target, file_size_mb=sample_file.stat().st_size / (1024 * 1024))
    print(f"   Detected problem: {analysis['problem_type']}")

    with session_scope() as db:
        ds = Dataset(
            name="Sample Linear Regression",
            original_filename="sample_linear_regression.csv",
            stored_path=str(sample_file.resolve()),
            file_extension=".csv",
            file_size_bytes=sample_file.stat().st_size,
            n_rows=len(frame),
            n_columns=len(frame.columns),
            target_column=target,
            problem_type=analysis["problem_type"],
            problem_type_source="agent",
            columns=list(frame.columns),
            column_dtypes={c: str(t) for c, t in frame.dtypes.items()},
            analysis=analysis,
            is_analyzed=True,
        )
        db.add(ds)
        db.flush()
        dataset_id = ds.id
        print(f"   Created dataset {dataset_id}")

        session = em.create_session(
            db=db,
            dataset_id=dataset_id,
            target_column=target,
            problem_type="regression",
            priority="speed",
            max_experiments=3,
            interactive=False,
            auto_start=False,
            name="E2E Regression Test",
        )
        session_id = session.id
        print(f"   Created session {session_id}")

    print("3. Starting session thread...")
    em.start_session_thread(session_id)

    print("4. Polling session progress...")
    for _ in range(60):
        prog = em.get_progress(session_id)
        print(f"   Status: {prog.get('status')} | Stage: {prog.get('stage')} | Message: {prog.get('message')}")
        if prog.get("status") in {"completed", "stopped", "failed"}:
            break
        time.sleep(1)

    print("5. Verifying DB records...")
    with session_scope() as db:
        session = db.get(ExperimentSession, session_id)
        print(f"   Final session status: {session.status}")
        print(f"   Stop reason: {session.stop_reason}")
        print(f"   Completed experiments: {session.completed_experiments}")
        print(f"   Recommendation available: {(session.recommendation or {}).get('available')}")
        print(f"   Recommended model: {(session.recommendation or {}).get('model_name')}")
        
        experiments = db.query(Experiment).filter(Experiment.session_id == session_id).all()
        print(f"   Number of experiment records: {len(experiments)}")
        for exp in experiments:
            print(f"     - #{exp.experiment_number}: {exp.model_name} ({exp.status}) -> metrics: {exp.metrics}")
            
        decisions = db.query(AgentDecision).filter(AgentDecision.session_id == session_id).all()
        print(f"   Number of decisions recorded: {len(decisions)}")
        for dec in decisions:
            print(f"     - Step {dec.step}: {dec.decision} | {dec.headline}")

    print("E2E Test completed successfully!")

if __name__ == "__main__":
    main()
