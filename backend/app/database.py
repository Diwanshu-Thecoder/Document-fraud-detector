"""
Phase 5: SQL-backed persistence.

SQLite via SQLAlchemy — no separate database server to install, the whole
thing is one file (app/fraud_detector.db), which keeps this easy to run
locally while still being real SQL (swapping to Postgres/MySQL later is
just a connection-string change, not a rewrite).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./app/fraud_detector.db"

# check_same_thread=False is needed because FastAPI can use a different
# thread per request; SQLite handles this fine for our access pattern
# (no concurrent writers).
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
